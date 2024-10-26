import json
import logging
import os
import subprocess
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np

from .config import GAEAConfig
from .utils import git, llm

logger = logging.getLogger("gaea")


def init_repo(config: GAEAConfig, origin="local", force_create=False) -> None:
    """Initialize the git repository

    Parameters
    ----------
    config
        The configuration object.
    origin
        "local" or "remote"
        When "local", the repository is initialized locally.
        When "remote", the repository is initialized by cloning from the remote repository.
    """
    git.create_git_dir(config.git_dir, force_create)
    if origin == "local":
        git.init_git_repo(config)
    elif origin == "remote":
        git.clone_git_repo(config)
    else:
        raise ValueError(f"Unknown option: {origin}")


def update_branches(config: GAEAConfig, pop: list[str]) -> None:
    """The branches are used to track the current state of the population"""
    branch_names = []
    hostname = config.hostname if config.hostname is not None else "host0"

    for index, _commit in enumerate(pop):
        branch_names.append(f"{hostname}-individual-{index}")

    git.branches_track_commits(config, branch_names, pop)


def get_initial_branches(config: GAEAConfig, pop_size: int) -> list[str]:
    """Get the initial branches with a simple strategy.
    1. Try to load the existing branches from the local repository.
        Existing branches are the branches that are already created from previous runs.
        This implies that we will try to restore from the previous state.
    2. If not enough, try to load the branches from the remote repository.
        Normally this happens when a new node joins the evolution.
        So it tries to kickstart the evolution by loading the branches from the remote.
    3. If still not enough, create new branches from the current head.
    """

    # Try to load the branches from the local repository
    branches = git.list_branches(config)
    pop = branches[:pop_size]
    pop = [git.get_commit_by_branch(config, branch) for branch in pop]

    # If not enough, try to load the branches from the remote repository
    if len(pop) < pop_size:
        remote_branches = git.list_branches(config, list_remote=True)
        remote_pop = remote_branches[: pop_size - len(pop)]
        remote_pop = [git.get_commit_by_branch(config, branch) for branch in remote_pop]
        pop.extend(remote_pop)

    # If still not enough, create new branches from the current head
    while len(pop) < pop_size:
        head = git.read_head_commit(config)
        pop.append(head)

    return pop


def push_local_branches(config: GAEAConfig) -> None:
    """Push all branches to the remote"""
    branches = git.list_branches(config)
    git.push_to_remote(config, branches)


def fetch_remote_branches(config: GAEAConfig) -> None:
    git.fetch_from_remote(config)


def prepare_temp_worktrees(config: GAEAConfig, commits: list[str]) -> None:
    worktrees = [git.add_temp_worktree(config, commit) for commit in commits]
    return worktrees


def cleanup_temp_worktrees(config: GAEAConfig) -> str:
    git.cleanup_temp_worktrees(config)


def evaluate(config: GAEAConfig, commit: str, worktree) -> dict[str, str]:
    """Evaluate the the result of the individual. This function can be run in parallel.

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
    if not config.reevaluate:
        # try to read the result from the git notes
        # if the result is found, directly return the result without evaluating
        note = git.read_note(config, commit)
        logger.info(f"Read note of {commit}: {note}\n")
        if note is not None:
            return json.loads(note)

    cmd = config.eval_command
    if config.enable_sandbox:
        # run the command in a bubblewrap sandbox
        cmd = [Path(__file__).parent / "bubblewrap_run.sh"] + cmd

    try:
        completed_proc = subprocess.run(
            cmd,
            capture_output=True,
            cwd=os.path.join(config.git_dir, worktree),
            timeout=config.timeout,
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

    logger.info(f"Evaluated {commit}\n")
    logger.info(f"Result: {note}\n")
    return result


def decode_result(output, illegal_value) -> Any:
    if output["timeout"] == True:
        fit = illegal_value
    else:
        try:
            stdout = json.loads(output["stdout"])
            if stdout["status"] == "finished":
                fit = stdout["fitness"]
            else:
                fit = illegal_value
        except json.JSONDecodeError:
            fit = illegal_value
        except Exception as e:
            logger.error(f"Unknown error occurred: {e}")
            fit = illegal_value

    return fit


def update_notes(
    config: GAEAConfig, commits: list[str], evaluate_results: list[str]
) -> None:
    for commit, result in zip(commits, evaluate_results):
        note = json.dumps(result)
        git.add_note(config, commit, note, overwrite=True)


def git_crossover(config: GAEAConfig, seed: int, commit1: str, commit2: str) -> str:
    """crossover between commit1 and commit2"""
    rng = np.random.default_rng(seed)
    use_merge = rng.choice([True, False], p=[config.merge_prob, 1 - config.merge_prob])
    if use_merge:
        git_merge(config, rng, commit1, commit2)
    else:
        git_rebase(config, rng, commit1, commit2)

    return git.read_head_commit(config)


def git_merge(
    config: GAEAConfig, rng: np.random.Generator, commit1: str, commit2: str
) -> None:
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

    assert not git.has_conflict(config)


def git_rebase(
    config: GAEAConfig, rng: np.random.Generator, commit1: str, commit2: str
) -> None:
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

    assert not git.has_conflict(config)


def prepare_llm_backend(llm_name: str, *args, **kwargs) -> Any:
    if "/" in llm_name:
        # e.g. meta-llama/Meta-Llama-3-8B-Instruct
        # which refers to a huggingface model
        return llm.HuggingfaceModel(llm_name, *args, **kwargs)
    elif llm_name.lower() == "gemini":
        return llm.GeminiBackend(*args, **kwargs)
    elif llm_name.lower() == "tgi":
        return llm.TGIBackend(*args, **kwargs)


CodeInfo = namedtuple("CodeInfo", ["code", "stack_trace", "timeout"])


def _construct_prompt(config, commits, chunk_size, operation_type) -> str:
    """A helper function to construct the prompt.
    Read the code and the related information (e.g. stack_trace) from the a list of commits.
    Then feed into the prompt constructor.
    The operation_type is either "mutation" or "crossover".
    The prompt_constructor is a function with the following signature:
    def prompt_constructor(operation_type: str, code_infos: List[CodeInfo]) -> str
    """
    code_infos = []
    for commit in commits:
        code = git.read_file(config, commit)
        note = git.read_note(config, commit)
        if note is not None:
            note = json.loads(note)
            if note["timeout"]:
                timeout = True
                stack_trace = None
            else:
                timeout = False
                try:
                    stack_trace = json.loads(note["stdout"])["stack_trace"]
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse the stack trace of {commit} with note: {note}"
                    )
                    stack_trace = None
                    import traceback

                    traceback.print_exc()

        code_infos.append(CodeInfo(code, stack_trace, timeout))

    prompts = []
    chunk = []
    for code_info in code_infos:
        chunk.append(code_info)
        if len(chunk) == chunk_size:
            prompts.append(config.prompt_constructor(operation_type, chunk))
            chunk = []

    return prompts


def llm_mutation(config, llm_backend, seeds, commits) -> list[str]:
    prompts = _construct_prompt(config, commits, 1, "mutation")

    responds = llm_backend.query(seeds, prompts)
    responds = [config.respond_extractor(response) for response in responds]
    offspring = []
    for commit, response in zip(commits, responds):
        code, commit_message = response
        git.update_file(config, commit, code, f"{config.llm_name}: {commit_message}")
        offspring.append(git.read_head_commit(config))

    return offspring


def llm_crossover(config, llm_backend, seeds, commits) -> list[str]:
    prompts = _construct_prompt(config, commits, 2, "crossover")

    responses = llm_backend.query(seeds, prompts)
    responses = [config.respond_extractor(response) for response in responses]
    offspring = []
    for commit, response in zip(commits, responses):
        code, commit_message = response
        git.update_file(config, commit, code, f"{config.llm_name}: {commit_message}")
        offspring.append(git.read_head_commit(config))

    return offspring


def migrate_from_human_tags(config: GAEAConfig, migrate_count: int) -> list[str]:
    """Return migration candidates from human tags. The migrate_count set the upper limit of the number of candidates."""
    all_tags = git.list_tags(config)
    tags = [tag for tag in all_tags if tag.startswith("human")]
    tags = tags[:migrate_count]
    commits = [git.get_commit_by_tag(config, tag) for tag in tags]
    git.delete_tags(config, tags)
    return commits


def migrate_from_other_hosts(config: GAEAConfig, migration_count: int) -> list[str]:
    """Return migration candidates from other hosts. The migration_count set the upper limit of the number of candidates."""
    remote_branches = git.list_branches(config, list_remote=True)
    hostname = config.hostname if config.hostname is not None else "host0"
    branches = [branch for branch in remote_branches if not branch.startswith(hostname)]
    import random

    random.shuffle(branches)
    branches = branches[:migration_count]
    commits = [git.get_commit_by_branch(config, branch) for branch in branches]
    notes = [git.read_note(config, commit) for commit in commits]
    fitness = [decode_result(json.loads(note), np.inf) for note in notes]
    fitness = np.array(fitness).astype(np.float32)
    return commits, fitness


def prune_commits(config: GAEAConfig) -> None:
    """Prune the commits that are reachable."""
    git.prune(config)
