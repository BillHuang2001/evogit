import json
import subprocess
import logging
from pathlib import Path

import numpy as np

from .config import GAEAConfig
from .utils import git, llm


def update_branches(config: GAEAConfig, pop: list[str]):
    """The branches are used to track the current state of the population"""
    branch_names = []
    hostname = config.hostname if config.hostname is not None else "host0"

    for index, _commit in enumerate(pop):
        branch_names.append(f"{hostname}-individual-{index}")

    git.branches_track_commits(config, branch_names, pop)


def get_initial_branches(config: GAEAConfig, pop_size: int):
    """Get the initial branches"""

    head = git.read_head_commit(config)
    return [head for _ in range(pop_size)]


def push_all_branches(config: GAEAConfig):
    """Push all branches to the remote"""
    pass


def evaluate(
    config: GAEAConfig, commit: str, cmd: list[str], timeout=300, sandbox=False
):
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
        note = git.read_note(config, commit)
        logger.info(f"Read note of {commit}: {note}\n")
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
    logger.info(f"Evaluated {commit}\n")
    logger.info(f"Result: {note}\n")
    return result


def git_crossover(config: GAEAConfig, seed: int, commit1: str, commit2: str):
    """crossover between commit1 and commit2"""
    rng = np.random.default_rng(seed)
    use_merge = rng.choice([True, False], p=[config.merge_prob, 1 - config.merge_prob])
    if use_merge:
        git_merge(config, rng, commit1, commit2)
    else:
        git_rebase(config, rng, commit1, commit2)

    return git.read_head_commit(config)


def git_merge(config: GAEAConfig, rng: np.random.Generator, commit1: str, commit2: str):
    git.checkout(config, commit1)
    git.merge_branches(config, commit2)

    if git.has_conflict(config):
        count = git.count_conflicts(config)
        strategy = rng.choice(
            [True, False],
            size=(count,),
            p=[config.accept_ours_prob, 1 - config.accept_ours_prob],
        )
        git.handle_conflict(config, strategy)
        git.continue_merge(config)


def git_rebase(
    config: GAEAConfig, rng: np.random.Generator, commit1: str, commit2: str
):
    git.checkout(config, commit1)
    git.rebase_branches(config, commit2)

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


def llm_mutation(config, llm_backend, seeds, commits):
    codes = [git.read_file(config, commit) for commit in commits]
    prompts = [mutation_prompt_constructor(code) for code in codes]
    responds = llm_backend.query(seeds, prompts)
    responds = [mutation_respond_extractor(response) for response in responds]
    offspring = []
    for commit, response in zip(commits, responds):
        git.update_file(config, commit, response, f"{config.llm_name}_mutation")
        offspring.append(git.read_head_commit(config))

    return offspring
