import json
import subprocess
import logging
from pathlib import Path

import numpy as np

from .config import GAEAConfig
from .utils import git, llm


def update_branches(config: GAEAConfig, pop: list[str]):
    """The branches are used to track the current state of the population"""
    git.update_tags(config, pop)
    branch_names = []
    hostname = config.hostname if config.hostname is not None else "host0"

    for index, tag in enumerate(pop):
        branch_names.append(f"{hostname}-individual-{index}")

    git.branches_track_tags(config, branch_names, pop)


def get_initial_branches(config: GAEAConfig, pop_size: int):
    """Get the initial branches"""
    tag = git.alloc_new_tag_name()
    git.tag(config, tag)

    return [tag for _ in range(pop_size)]


def push_all_branches(config: GAEAConfig):
    """Push all branches to the remote"""
    pass


def evaluate(config: GAEAConfig, tag: str, cmd: list[str], timeout=300, sandbox=False):
    """Evaluate the the result of the individual.

    Parameters
    ----------
    codegen_dir
        The directory of the codegen.
    branch
        The branch name.
    cmd
        List of strings. The command to run the code.
        For example ["python", "main.py"]
        The command will be run with the current working directory set to the branch directory.
    handler
        A function that takes the result of the command and return the fitness.
        The result is a dictionary of the following: {
            "stdout": str,
            "stderr": str,
            "timeout": bool
        }
    """
    logger = logging.getLogger("main")
    if not config.reevaluate:
        # try to read the result from the git notes
        # if the result is found, directly return the result without evaluating
        note = git.read_note(config, tag)
        logger.info(f"Read note of {tag}: {note}\n")
        if note is not None:
            return json.loads(note)

    if sandbox:
        # run the command in a bubblewrap sandbox
        cmd = [Path(__file__).parent / "bubblewrap_run.sh"] + cmd

    try:
        completed_proc = subprocess.run(
            cmd,
            capture_output=True,
            cwd=config.git_dir,
            timeout=timeout,
        )
        stdout = completed_proc.stdout.decode("utf-8")
        stderr = completed_proc.stderr.decode("utf-8")
        result = {
            "stdout": stdout,
            "stderr": stderr,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        result = {
            "stdout": "",
            "stderr": "",
            "timeout": True,
        }

    note = json.dumps(result)
    git.add_note(config, note, overwrite=True)
    logger.info(f"Evaluated {tag}\n")
    logger.info(f"Result: {note}\n")
    return result


def git_crossover(config: GAEAConfig, seed: int, tag1: str, tag2: str):
    """crossover between tag1 and tag2"""
    rng = np.random.default_rng(seed)
    use_merge = rng.choice([True, False], p=[config.merge_prob, 1 - config.merge_prob])
    if use_merge:
        git_merge(config, rng, tag1, tag2)
    else:
        git_rebase(config, rng, tag1, tag2)

    new_tag = git.alloc_new_tag_name()
    git.tag(config, new_tag)

    return new_tag


def git_merge(config: GAEAConfig, rng: np.random.Generator, tag1: str, tag2: str):
    git.checkout(config, tag1)
    git.merge_branches(config, tag2)

    if git.has_conflict(config):
        count = git.count_conflicts(config)
        strategy = rng.choice(
            [True, False],
            size=(count,),
            p=[config.accept_ours_prob, 1 - config.accept_ours_prob],
        )
        git.handle_conflict(config, strategy)
        git.continue_merge(config)


def git_rebase(config: GAEAConfig, rng: np.random.Generator, tag1: str, tag2: str):
    git.checkout(config, tag1)
    git.rebase_branches(config, tag2)

    while git.has_conflict(config):
        count = git.count_conflicts(config)
        strategy = rng.choice(
            [True, False],
            size=(count,),
            p=[config.accept_ours_prob, 1 - config.accept_ours_prob],
        )
        git.handle_conflict(config, strategy)
        git.continue_rebase(config)


normal_mutation_template = """Your task is to write an algorithm that solves the Travelling Salesman Problem (TSP).
The solution is encoded as a permutation of the indices of the cities.
The code should be written in EvoX, which uses JAX and Python.
In `setup` the algorithm can generate the initial population.
In `ask` the algorithm can return the offspring and the update state.
In `tell` the algorithm can update the state.
The offspring return by the `ask` will be evaluated by another program, so you do not need to implement the evaluation.
The state can be used to store mutable variables across `ask` and `tell`, for example, you can store the offspring in the state in `ask` and uses it latter on in `tell`.
You can also add more functions or methods if needed.
Here is your original code:
```
{}
```
Please try to improve it. You can modify either the `setup`, `ask`, or `tell` function or add new functions or methods.
If there are bugs, please fix them as well.
The result should be given in between ``` and ```, do not explain.
"""


def mutation_prompt_constructor(code):

    return normal_mutation_template.format(code)


def mutation_respond_extractor(response):
    try:
        return response.split("```")[1].strip() + "\n"
    except IndexError:
        return ""


def prepare_llm_backend(config: GAEAConfig):
    if "/" in config.llm_name:
        # e.g. meta-llama/Meta-Llama-3-8B-Instruct
        # which refers to a huggingface model
        return llm.HuggingfaceModel(config.llm_name, config.device_map)
    elif config.llm_name.lower() == "gemini":
        return llm.GeminiBackend(config.api_key, config.http_req_params)


def llm_mutation(config, llm_backend, seeds, tags):
    codes = [git.read_file(config, tag) for tag in tags]
    prompts = [mutation_prompt_constructor(code) for code in codes]
    responds = llm_backend.query(seeds, prompts)
    responds = [mutation_respond_extractor(response) for response in responds]
    offspring = []
    for tag, response in zip(tags, responds):
        git.update_file(config, tag, response, f"{config.llm_name}_mutation")
        new_tag = git.alloc_new_tag_name()
        git.tag(config, new_tag)
        offspring.append(new_tag)

    return offspring
