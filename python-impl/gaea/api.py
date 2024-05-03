import json
import subprocess

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


def evaluate(config: GAEAConfig, tag: str, cmd: list[str], handler: callable, timeout=60, sandbox=True):
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
    if sandbox:
        # run the command in a bubblewrap sandbox
        cmd = ["bubblewrap_run.sh"] + cmd

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
    
    fitness = handler(result)

    note = json.dump(result)
    git.add_note(config, note)
    return fitness


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


def mutation_prompt_constructor(code):
    normal_mutation_template = """
    Your task is to write a function that calculate pi.
    Here is your original code:
    ```
    {}
    ```
    Please try to improve it.
    The result should be given in between ``` and ```, do not explain.
    """

    fix_mutation_template = """
    Here is your original code:
    ```
    {}
    ```
    The code is incorrect.
    And the error is:
    ```
    ```
    Please try to fix it.
    """

    return normal_mutation_template.format(code)


def mutation_respond_extractor(response):
    return response.split("```")[1].strip() + "\n"


def prepare_llm_backend(config: GAEAConfig):
    llm.HuggingfaceModel(config.name, config.device_map)


def llm_mutation(config, llm_backend, seeds, tags):
    codes = [git.read_file(config, tag) for tag in tags]
    prompts = [mutation_prompt_constructor(code) for code in codes]
    responds = llm_backend.query(seeds, prompts)
    responds = [mutation_respond_extractor(response) for response in responds]
    offspring = []
    for tag, response in zip(tags, responds):
        git.update_file(config, tag, response, f"{config.llm_names[0]}_mutation")
        new_tag = git.alloc_new_tag_name()
        git.tag(config, new_tag)
        offspring.append(new_tag)

    return offspring
