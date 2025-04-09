import json
import logging
import io
import os
import subprocess
from collections import namedtuple
from pathlib import Path
from typing import Any, Tuple
import random

import numpy as np

from .config import PhyloXConfig
from .utils import git, llm, prompt

logger = logging.getLogger("phylox")


def init_repo(config: PhyloXConfig, origin="local", force_create=False) -> None:
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


def update_branches(config: PhyloXConfig, pop: list[str]) -> None:
    """The branches are used to track the current state of the population"""
    branch_names = []
    hostname = config.hostname if config.hostname is not None else "host0"

    for index, _commit in enumerate(pop):
        branch_names.append(f"{hostname}-individual-{index}")

    git.branches_track_commits(config, branch_names, pop)


def get_initial_branches(config: PhyloXConfig, pop_size: int) -> list[str]:
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


def push_local_branches(
    config: PhyloXConfig,
) -> Tuple[subprocess.Popen | None, subprocess.Popen | None]:
    """Push all branches to the remote"""
    branches = git.list_branches(config)
    proc1 = git.push_notes_to_remote(config)
    proc2 = git.push_to_remote(config, branches)
    return proc1, proc2


def fetch_remote(
    config: PhyloXConfig,
) -> Tuple[subprocess.Popen | None, subprocess.Popen | None]:
    """Fetch remote branches and notes"""
    proc1 = git.fetch_from_remote(config)
    proc2 = git.fetch_notes_from_remote(config)
    return proc1, proc2


def prepare_temp_worktrees(config: PhyloXConfig, commits: list[str]) -> list[str]:
    worktrees = [git.add_temp_worktree(config, commit) for commit in commits]
    return worktrees


def cleanup_temp_worktrees(config: PhyloXConfig) -> None:
    git.cleanup_temp_worktrees(config)


def evaluate_code(config: PhyloXConfig, commit: str, worktree) -> dict[str, str]:
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
    return result


def decode_result(output, illegal_value) -> Any:
    if output["timeout"] is True:
        performance_cost = illegal_value
        time_cost = illegal_value
    else:
        try:
            stdout = json.loads(output["stdout"])
            if stdout["status"] == "finished":
                performance_cost = stdout["fitness"]
                time_cost = stdout["time_cost"]
            else:
                performance_cost = illegal_value
                time_cost = illegal_value
        except json.JSONDecodeError:
            performance_cost = illegal_value
            time_cost = illegal_value
        except Exception as e:
            logger.error(f"Unknown error occurred: {e}")
            performance_cost = illegal_value
            time_cost = illegal_value

    return performance_cost, time_cost


def update_notes(
    config: PhyloXConfig, commits: list[str], evaluate_results: list[str]
) -> None:
    for commit, result in zip(commits, evaluate_results):
        note = json.dumps(result)
        git.add_note(config, commit, note, overwrite=True)


def is_novel_merge(config: PhyloXConfig, commit1: str, commit2: str) -> bool:
    """Check if the merge of commit1 and commit2 is novel, that is, A and B are not ancestor of each other."""
    return not (
        git.fast_forwardness(config, commit1, commit2)
        or git.fast_forwardness(config, commit2, commit1)
    )


def git_crossover(config: PhyloXConfig, seed: int, commit1: str, commit2: str) -> str:
    """crossover between commit1 and commit2"""
    random.seed(seed)
    use_merge = random.choices(
        [True, False], weights=[config.merge_prob, 1 - config.merge_prob]
    )[0]
    if use_merge:
        git_merge(config, commit1, commit2)
    else:
        git_rebase(config, commit1, commit2)

    return git.read_head_commit(config)


def git_merge(config: PhyloXConfig, commit1: str, commit2: str) -> None:
    git.checkout(config, commit1)
    git.merge_branches(config, commit2)

    if git.has_conflict(config):
        count = git.count_conflicts(config)
        strategy = random.choices(
            [True, False],
            weights=[config.accept_ours_prob, 1 - config.accept_ours_prob],
            k=count,
        )
        git.handle_conflict(config, strategy)
        git.continue_merge(config)

    assert not git.has_conflict(config)


def git_rebase(config: PhyloXConfig, commit1: str, commit2: str) -> None:
    git.checkout(config, commit1)
    git.rebase_branches(config, commit2)

    while git.has_conflict(config):
        count = git.count_conflicts(config)
        strategy = random.choices(
            [True, False],
            weights=[config.accept_ours_prob, 1 - config.accept_ours_prob],
            k=count,
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


def _gather_info(config, commits) -> list[dict[str, Any]]:
    worktrees = prepare_temp_worktrees(config, commits)
    infos = []
    for commit, worktree in zip(commits, worktrees):
        file_list = git.list_files(config, commit)
        files = [file for file in file_list.split("\n") if file != ""]
        text_file_extensions = ["js", "jsx", "ts", "tsx", "html", "css", "scss", "json"]
        text_files = [
            file for file in files if file.split(".")[-1] in text_file_extensions
        ]
        random_file = random.choice(text_files)
        random_file_path = os.path.join(worktree, random_file)
        with open(random_file_path, "r") as f:
            code = f.read()
        code = code.split("\n")
        n_lines = len(code)
        random_section_start = random.randint(0, n_lines - 1)
        random_section_end = random.randint(
            random_section_start, min(random_section_start + 50, n_lines)
        )
        infos.append(
            {
                "commit": commit,
                "worktree": worktree,
                "file_list": file_list,
                "random_file": random_file,
                "code": code,
                "random_section_start": random_section_start,
                "random_section_end": random_section_end,
                "n_lines": n_lines,
            }
        )

    return infos


def llm_constrained_mutation(config, llm_backend, seeds, commits) -> list[str]:
    """Randomly select a file and a section within the file and let the llm to mutate the code."""
    infos = _gather_info(config, commits)
    prompts = []
    for info in infos:
        # insert <|EDIT|> and <|END_EDIT|> to the code
        # insert end first, so that the indices are not changed
        prompt_code = info["code"][:]
        prompt_code.insert(info["random_section_end"], "<|END_EDIT|>")
        prompt_code.insert(info["random_section_start"], "<|EDIT|>")
        prompt_code = "\n".join(prompt_code)
        prompt_text = config.prompt_constructor(info["file_list"], prompt_code)
        prompts.append(prompt_text)

    responses = llm_backend.query(seeds, prompts)
    responses = [config.respond_extractor(response) for response in responses]
    code_changes = [responses[0] for responses in responses]
    commit_messages = [responses[1] for responses in responses]
    # update the code with the code changes
    edited_codes = []
    for info, code_change, commit_message in zip(infos, code_changes, commit_messages):
        edited_code = info["code"][:]
        start = info["random_section_start"]
        end = info["random_section_end"]
        edited_code[start:end] = [code_change]
        edited_code = "\n".join(edited_code)
        git.update_file(
            config,
            info["commit"],
            edited_code,
            f"{config.llm_name}: {commit_message}",
        )
        edited_codes.append(git.read_head_commit(config))

    cleanup_temp_worktrees(config)
    return edited_codes


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


def load_vectors(config: PhyloXConfig, commits: list[str]) -> np.ndarray:
    vectors = []
    for commit in commits:
        binary_data = git.read_file(config, commit, mode="binary")
        with io.BytesIO(binary_data) as f:
            vector = np.load(f)
            vectors.append(vector)

    return np.array(vectors)


def vector_mutation(
    config: PhyloXConfig, commits: list[str], mutation_func: callable
) -> str:
    """Mutation of the individual"""
    vectors = load_vectors(config, commits)
    mutated_vectors = mutation_func(vectors)
    offspring = []
    for commit, vector in zip(commits, mutated_vectors):
        with io.BytesIO() as f:
            np.save(f, vector)
            git.update_file(config, commit, f.getvalue(), "Mutation")
            offspring.append(git.read_head_commit(config))

    return offspring


def vector_direct_crossover(config: PhyloXConfig, seed: int, commits: list[str]) -> str:
    vectors = []
    for commit in commits:
        binary_data = git.read_file(config, commit, mode="binary")
        with io.BytesIO(binary_data) as f:
            vector = np.load(f)
            vectors.append(vector)

    vectors = np.array(vectors)


def migrate_from_human_tags(config: PhyloXConfig, migrate_count: int) -> list[str]:
    """Return migration candidates from human tags. The migrate_count set the upper limit of the number of candidates."""
    all_tags = git.list_tags(config)
    tags = [tag for tag in all_tags if tag.startswith("human")]
    tags = tags[:migrate_count]
    commits = [git.get_commit_by_tag(config, tag) for tag in tags]
    git.delete_tags(config, tags)
    return commits


def migrate_from_other_hosts(config: PhyloXConfig, migration_count: int) -> list[str]:
    """Return migration candidates from other hosts. The migration_count set the upper limit of the number of candidates."""
    remote_branches = git.list_branches(config, list_remote=True)
    git.merge_notes(config)
    hostname = config.hostname if config.hostname is not None else "host0"
    # filter out the branches that are from the current host
    branches = [branch for branch in remote_branches if hostname not in branch]

    random.shuffle(branches)
    commits = []
    fitness = []
    for branch in branches:
        commit = git.get_commit_by_branch(config, branch)
        note = git.read_note(config, commit)
        if note is not None:
            commits.append(commit)
            fitness.append(decode_result(json.loads(note), np.inf))
        else:
            logger.warning(f"Note is not found for branch: {branch}  id: {commit}")

        if len(commits) == migration_count:
            break

    fitness = np.array(fitness).astype(np.float32)
    return commits, fitness


def prune_commits(config: PhyloXConfig) -> None:
    """Prune the commits that are reachable."""
    git.prune(config)


def _construct_diff_comp_prompt(config, prev_commit, new_commit) -> str:
    """A helper function to construct the prompt.
    Read the code and the related information (e.g. stack_trace) from the a list of commits.
    Then feed into the prompt constructor.
    The operation_type is either "mutation" or "crossover".
    The prompt_constructor is a function with the following signature:
    def prompt_constructor(operation_type: str, code_infos: List[CodeInfo]) -> str
    """
    diff = git.diff_view(config, prev_commit, new_commit)
    prev_file_list = git.list_files(config, prev_commit)
    prev_info = git.read_note(config, prev_commit)
    new_info = git.read_note(config, new_commit)
    prompt = config.diff_prompt_constructor(prev_file_list, diff, prev_info, new_info)
    return prompt


def llm_diff_compare(config: PhyloXConfig, prev_commit: str, new_commit: str) -> bool:
    """Return True if the change from prev_commit to new_commit is good, and False otherwise"""
    prompt = _construct_diff_comp_prompt(config, prev_commit, new_commit)
    response = llm.query(prompt)
    if "good" in response.lower():
        return True
    elif "bad" in response.lower():
        return False
    else:
        logger.warning(
            f"Unknown response from LLM: {response}. " "Assuming the change is bad."
        )
        return False
