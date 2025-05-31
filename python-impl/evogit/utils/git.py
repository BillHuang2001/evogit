"""
The utility functions for git operations.
"""

import subprocess
import warnings
import os
import shutil
import tempfile
from phylox.config import EvoGitConfig
from typing import Optional, Union
import re


# used to match the conflict pattern in the file
# example:
# <<<<<<< HEAD
# content in HEAD
# =======
# content in branch_name
# >>>>>>> branch_name
# the first group matches the content in HEAD, and the second group matches the content in branch_name
git_conflict_pattern = re.compile(
    r"<<<<<<<.*?\n(.*?)=======.*?\n(.*?)>>>>>>>.*?\n", re.DOTALL
)
# sometimes the LLM can output null characters, which will cause git commit to fail
# make sure it only contain alphanumeric, space, underscore and hyphen
git_commit_message_pattern = re.compile(r"[^a-zA-Z0-9:_\-\./\\ ]")


def create_git_dir(git_dir, force_create=False) -> None:
    if os.path.exists(git_dir):
        print(f"{git_dir} already exists! Do you want to remove it? (y/n)")
        if force_create or input() == "y":
            print(f"Removing {git_dir}...")
            for filename in os.listdir(git_dir):
                file_path = os.path.join(git_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    raise ValueError(f"Unknown file type: {file_path}")
        else:
            print("Abort!")
            exit()
    else:
        os.mkdir(git_dir, mode=0o755)


def delete_remote_branches(config: EvoGitConfig) -> None:
    fetch_from_remote(config, async_fetch=False)
    remote_branches = list_branches(config, list_remote=True)
    print(f"Delete the following remote branches: {remote_branches}")

    # delete all remote branches
    branch_names = []
    for remote_branch in remote_branches:
        branch_name = remote_branch.split("/")[-1]
        branch_names.append(branch_name)

    if branch_names:
        subprocess.run(
            ["git", "push", "-q", "-d", "origin"] + branch_names,
            cwd=config.git_dir,
            check=True,
        )


def delete_remote_notes(config: EvoGitConfig) -> None:
    fetch_notes_from_remote(config, async_fetch=False)
    remote_notes_namespaces = list_remote_notes_namespaces(config)
    print(f"Delete the following remote notes: {remote_notes_namespaces}")

    # delete all remote notes
    for remote_notes_namespace in remote_notes_namespaces:
        subprocess.run(
            [
                "git",
                "push",
                "-q",
                "origin",
                "-d",
                f"refs/notes/{remote_notes_namespace}",
            ],
            cwd=config.git_dir,
            check=True,
        )


def init_git_repo(config: EvoGitConfig) -> None:
    git_dir = config.git_dir

    subprocess.run(
        ["git", "init", "--object-format", config.git_hash], cwd=git_dir, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", config.git_user_name], cwd=git_dir, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", config.git_user_email], cwd=git_dir, check=True
    )
    if os.path.isdir(config.seed_file):
        shutil.copytree(config.seed_file, git_dir)
    else:
        shutil.copy(config.seed_file, os.path.join(git_dir, config.filename))

    subprocess.run(["git", "add", "."], cwd=git_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=git_dir, check=True)
    # disable auto gc, since we will run it manually at the end of each evaluation
    subprocess.run(["git", "config", "gc.auto", "0"], cwd=git_dir, check=True)
    config_merge_driver(config)

    if config.remote_repo is not None:
        subprocess.run(
            ["git", "remote", "add", "origin", config.remote_repo],
            cwd=git_dir,
            check=True,
        )
        subprocess.run(
            ["git", "push", "-f", "origin", "master"], cwd=git_dir, check=True
        )
        subprocess.run(
            ["git", "push", "-d", "origin", "refs/notes/commits"],
            cwd=git_dir,
            check=True,
        )

        delete_remote_branches(config)
        delete_remote_notes(config)


def clone_git_repo(config: EvoGitConfig) -> None:
    if config.remote_repo is None:
        raise ValueError("remote_repo is not set in the config.")

    git_dir = config.git_dir

    subprocess.run(
        ["git", "clone", config.remote_repo, git_dir],
        cwd=config.git_dir,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", config.git_user_name], cwd=git_dir, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", config.git_user_email], cwd=git_dir, check=True
    )
    # disable auto gc, since we will run it manually at the end of each evaluation
    subprocess.run(["git", "config", "gc.auto", "0"], cwd=git_dir, check=True)
    config_merge_driver(config)


def config_merge_driver(config: EvoGitConfig) -> None:
    template = (
        '[merge "{name}"]\n'
        "\tname = phylox custom merge driver\n"
        "\tdriver = {path} %O %A %B\n"
    )
    if config.merge_driver is not None:
        with open(os.path.join(config.git_dir, ".git/config"), "a") as f:
            f.write(
                template.format(
                    name="phylox-custom-merge-driver", path=config.merge_driver
                )
            )

        with open(os.path.join(config.git_dir, ".gitattributes"), "a") as f:
            f.write("*.npy merge=phylox-custom-merge-driver\n")


def get_commit_by_branch(
    config: EvoGitConfig, branch: str, worktree: Optional[str] = None
) -> str:
    cwd = worktree if worktree is not None else config.git_dir
    return (
        subprocess.run(
            ["git", "rev-parse", branch],
            cwd=cwd,
            check=True,
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .strip()
    )


def get_commit_by_tag(
    config: EvoGitConfig, tag: str, worktree: Optional[str] = None
) -> str:
    return get_commit_by_branch(config, f"tags/{tag}", worktree)


def read_head_commit(config: EvoGitConfig, worktree: Optional[str] = None) -> str:
    """Read the commit id of the current HEAD."""
    return get_commit_by_branch(config, "HEAD", worktree)


def list_branches(
    config: EvoGitConfig, list_remote: bool = False, ea_only: bool = True
) -> list[str]:
    """List all branches in this repo.
    Parameters
    ----------
    list_remote:
        If True, list the remote branches. Otherwise, list the local branches.
    ea_only
        If True, only return the branches that are related to the EA.
        Master, main and detached head branches will be excluded.
    """
    cmd = ["git", "branch", "--no-color"]
    if list_remote:
        cmd.append("-r")

    branches = subprocess.run(
        cmd,
        cwd=config.git_dir,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    branches = branches.split("\n")
    branches = [branch.strip() for branch in branches if branch != ""]
    # the current branch is marked with a "* "
    # the current branch in other worktrees is marked with a "+ "
    # find and remove the "* ", "+ "
    for i, branch in enumerate(branches):
        if branch.startswith("* "):
            branches[i] = branch[2:]
        elif branch.startswith("+ "):
            branches[i] = branch[2:]

    def is_normal_branch(branch):
        return not (
            "detached"
            in branch  # detached head is a result of unclean exit from previous runs
            or "master" in branch  # master is not used by the EA
            or "main" in branch  # main is not used by the EA
            or "HEAD" in branch  # HEAD or remote/origin/HEAD should not be used
        )

    if ea_only:
        branches = [b for b in branches if is_normal_branch(b)]

    if list_remote:
        branches = ["remotes/" + b for b in branches]

    return branches


def list_tags(config: EvoGitConfig) -> list[str]:
    """List all tags in this repo."""
    tags = subprocess.run(
        ["git", "tag"],
        cwd=config.git_dir,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    tags = tags.split("\n")
    tags = [tag.strip() for tag in tags if tag != ""]
    return tags


def delete_tags(config: EvoGitConfig, tags: list[str]) -> None:
    """Delete the specified tags."""
    if tags:
        subprocess.run(
            ["git", "tag", "-d"] + tags,
            cwd=config.git_dir,
            check=True,
        )


def add_note(
    config: EvoGitConfig, commit: str, note: str | bytes, overwrite: bool = False
) -> None:
    """Add a note to the current commit. If overwrite is True, force overwrite the existing note."""
    # the note is passed through stdin, so we use `-F -` to let git read from stdin
    # otherwise, if we use `-m`, and the note is too long, it will result in an error
    cmd = ["git", "notes", "add", "-F", "-"]
    if overwrite:
        cmd.append("-f")

    cmd.append(commit)

    if isinstance(note, str):
        note = note.encode("utf-8")

    subprocess.run(
        cmd,
        cwd=config.git_dir,
        input=note,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def append_note(config: EvoGitConfig, commit: str, note: str | bytes) -> None:
    """Append a note to the current commit."""
    if isinstance(note, str):
        note = note.encode("utf-8")

    subprocess.run(
        ["git", "notes", "append", "-"],
        cwd=config.git_dir,
        input=note,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def read_note(config: EvoGitConfig, commit: Optional[str]) -> Optional[str]:
    """Read the note of the specified commit. If commit is None, read the note of the current HEAD.
    Return None if the note does not exist.
    """

    cmd = ["git", "notes", "show"]
    if commit is not None:
        cmd.append(commit)

    completed_proc = subprocess.run(
        cmd,
        cwd=config.git_dir,
        capture_output=True,
    )

    if completed_proc.returncode != 0:
        return None
    else:
        return completed_proc.stdout.decode("utf-8")


def update_file_in_worktree(
    config: EvoGitConfig,
    worktree: str,
    new_content: Union[str, bytes],
    filename: Optional[str] = None,
) -> None:
    """Update the content of the file in the specified worktree, then commit the updated file."""
    filename = filename if filename is not None else config.filename
    if isinstance(new_content, str):
        mode = "r+"
    elif isinstance(new_content, bytes):
        mode = "rb+"
    else:
        raise ValueError("new_content must be either a string or bytes.")

    with open(os.path.join(worktree, filename), mode) as f:
        current_content = f.read()
        # if the content is the same, do nothing
        if current_content == new_content:
            return

        f.seek(0)
        f.write(new_content)
        f.truncate()

    subprocess.run(["git", "add", filename], cwd=worktree, check=True)


def add_file_in_worktree(
    config: EvoGitConfig,
    worktree: str,
    new_content: Union[str, bytes],
    file_path: str,
):
    if isinstance(new_content, str):
        mode = "w"
    elif isinstance(new_content, bytes):
        mode = "wb"
    else:
        raise ValueError("new_content must be either a string or bytes.")

    full_path = os.path.join(worktree, file_path)
    if os.path.exists(full_path):
        warnings.warn(f"{file_path} already exists!")
    else:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, mode) as f:
            f.write(new_content)

    subprocess.run(["git", "add", file_path], cwd=worktree, check=True)


def commit_changes_in_worktree(
    config: EvoGitConfig,
    worktree: str,
    commit_message: str,
):
    """Commit the changes in the specified worktree. Assume that the changes are already staged."""
    if not has_staged_changes(config, worktree):
        warnings.warn(
            f"No staged changes in {worktree}. Skipping commit."
        )
        return

    commit_message = git_commit_message_pattern.sub("", commit_message)
    commit_message = commit_message[:256]  # truncate the message to 256 characters
    n_retry = 0
    while n_retry < 3:
        n_retry += 1
        # git commit could fail if no files are staged or due to some race condition
        proc = subprocess.run(
            ["git", "commit", "-q", "-m", commit_message],
            cwd=worktree,
            capture_output=True,
        )
        if proc.returncode != 0:
            stdout = proc.stdout.decode("utf-8")
            stderr = proc.stderr.decode("utf-8")
            warnings.warn(
                f"Failed to commit changes in {worktree}. Status: {proc.returncode}, stdout: {stdout} stderr: {stderr}"
            )
        else:
            break


def update_file(
    config: EvoGitConfig,
    commit: str,
    new_content: Union[str, bytes],
    commit_message: str,
    filename: Optional[str] = None,
) -> None:
    """Update the content of the file in the specified commit_id and commit the updated file.
    new_content can be either a string or bytes,
    when it is a string, it will be written in text mode, otherwise, it will be written in binary mode.
    """
    checkout(config, commit)

    filename = filename if filename is not None else config.filename

    if isinstance(new_content, str):
        mode = "r+"
    elif isinstance(new_content, bytes):
        mode = "rb+"
    else:
        raise ValueError("new_content must be either a string or bytes.")

    with open(os.path.join(config.git_dir, filename), mode) as f:
        current_content = f.read()
        # if the content is the same, do nothing
        if current_content == new_content:
            return

        f.seek(0)
        f.write(new_content)
        f.truncate()

    subprocess.run(["git", "add", filename], cwd=config.git_dir, check=True)
    commit_message = git_commit_message_pattern.sub("", commit_message)
    commit_message = commit_message[:256]  # truncate the message to 256 characters
    subprocess.run(
        ["git", "commit", "-q", "-m", commit_message], cwd=config.git_dir, check=True
    )


def read_file(config: EvoGitConfig, commit: str, mode: str = "text") -> str | bytes:
    """Read the content of the file in the specified commit."""
    completed_proc = subprocess.run(
        ["git", "show", f"{commit}:{config.filename}"],
        cwd=config.git_dir,
        check=True,
        capture_output=True,
    )

    if mode == "text":
        return completed_proc.stdout.decode("utf-8")
    elif mode == "binary":
        return completed_proc.stdout
    else:
        raise ValueError("mode must be either 'text' or 'binary'.")


def batch_read_files(config: EvoGitConfig, commits: list[str]) -> list[str]:
    return [read_file(config, commit) for commit in commits]


def has_conflict(config: EvoGitConfig) -> bool:
    """Return True if the current working directory has conflicts. Otherwise, return False."""
    status = subprocess.run(
        ["git", "status"], cwd=config.git_dir, capture_output=True, check=True
    ).stdout.decode("utf-8")

    if "Unmerged paths" in status or "rebasing" in status or "merging" in status:
        return True
    else:
        return False


def count_conflicts(config: EvoGitConfig, filename: Optional[str] = None) -> int:
    """Count the number of conflicts in the current working directory."""
    if filename is None:
        filename = config.filename

    with open(os.path.join(config.git_dir, filename), "r") as f:
        content = f.read()

    return len(git_conflict_pattern.findall(content))


def list_conflict_files(config: EvoGitConfig) -> list[str]:
    """List all the files that have conflicts in the current working directory."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=config.git_dir,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")

    conflict_files = [file for file in result.split("\n") if file != ""]

    return conflict_files


def checkout(config: EvoGitConfig, commit: str) -> None:
    """Checkout the specified commit."""
    # -q is quiet, --detach is used to checkout the commit in detached HEAD mode
    subprocess.run(
        ["git", "checkout", "-q", "--detach", commit],
        cwd=config.git_dir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def add_temp_worktree(config: EvoGitConfig, branch: str) -> str:
    """checkout the branch in a new worktree and return the path of the worktree."""
    # make a temporary directory if not exists
    worktree_dir = os.path.join(config.git_dir, ".phylox_evaluate")
    if not os.path.exists(worktree_dir):
        # in case we have multiple instances running at the same time
        try:
            os.mkdir(worktree_dir)
        except FileExistsError:
            pass

    worktree = tempfile.mkdtemp(prefix=f"{branch}_", dir=worktree_dir)
    subprocess.run(
        ["git", "worktree", "add", "--detach", "-q", worktree, branch],
        cwd=config.git_dir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return worktree


def remove_temp_worktree(config: EvoGitConfig, worktree: str) -> None:
    """Remove the worktree of the branch."""
    subprocess.run(
        ["git", "worktree", "remove", "-f", worktree],
        cwd=config.git_dir,
        check=True,
    )


def cleanup_temp_worktrees(config: EvoGitConfig) -> None:
    """Remove all the worktrees in the .phylox_evaluate directory."""
    worktree_dir = os.path.join(config.git_dir, ".phylox_evaluate")
    if os.path.exists(worktree_dir):
        shutil.rmtree(worktree_dir)

    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=config.git_dir,
        check=True,
    )


def merge_branches(config: EvoGitConfig, commit: str) -> None:
    """merge the commit specified by the commit_id to the current branch."""
    # don't check the return code because the merge may fail
    subprocess.run(
        ["git", "merge", "-q", commit, "--no-edit"],
        cwd=config.git_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def rebase_branches(config: EvoGitConfig, commit: str) -> None:
    """rebase the current branch on the commit specified by the commit_id."""
    # don't check the return code because the rebase may fail
    # GIT_EDITOR=true is used to disable the interactive editor
    # and accept the default commit message
    subprocess.run(
        ["git", "rebase", "-q", commit],
        cwd=config.git_dir,
        env=os.environ | {"GIT_EDITOR": "true"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def continue_merge(config: EvoGitConfig) -> None:
    """Continue the merge process."""
    subprocess.run(
        ["git", "merge", "--continue"],
        cwd=config.git_dir,
        env=os.environ | {"GIT_EDITOR": "true"},
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def continue_rebase(config: EvoGitConfig) -> None:
    """Continue the rebase process."""
    # GIT_EDITOR=true is used to disable the interactive editor
    # and accept the default commit message
    subprocess.run(
        ["git", "rebase", "--continue"],
        cwd=config.git_dir,
        env=os.environ | {"GIT_EDITOR": "true"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def handle_conflict(
    config: EvoGitConfig, strategy: list[bool], filename: Optional[str] = None
) -> None:
    """Handle the conflict by accept ours or theirs.
    The strategy is a list of bool values, where True means accepting ours, and False means accepting theirs.
    Write back the result to the file and git add.
    """
    if filename is None:
        filename = config.filename

    with open(os.path.join(config.git_dir, filename), "r") as f:
        content = f.read()

    iterator = iter(strategy)

    def handle_one_conflict(match):
        accept_ours = next(iterator)
        if accept_ours:
            return match.group(1)
        else:
            return match.group(2)

    result = git_conflict_pattern.sub(handle_one_conflict, content)
    with open(os.path.join(config.git_dir, filename), "w") as f:
        f.write(result)

    subprocess.run(["git", "add", filename], cwd=config.git_dir, check=True)


def branches_track_commits(
    config: EvoGitConfig, branch_names: list[str], commits: list[str]
) -> None:
    """Create branches that track the specified commits."""
    processes = []
    for branch_name, commit in zip(branch_names, commits):
        proc = subprocess.Popen(
            ["git", "branch", "-f", branch_name, commit],
            cwd=config.git_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(proc)

    for proc in processes:
        proc.wait()
        assert proc.returncode == 0, "Failed to create branch that tracks the commit."


def fast_forwardness(config: EvoGitConfig, commit1: str, commit2: str) -> bool:
    """Check if commit1 is able to fast-forward to commit2. Return True if it is able to fast-forward."""
    code = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit1, commit2],
        cwd=config.git_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    return code == 0


def pairwise_distances(config: EvoGitConfig, commits: list[str]) -> list[list[int]]:
    """Calculate the pairwise distances between the commits.
    Return a matrix where the element at (i, j) is the distance between commit i and commit j.
    The matrix is guaranteed to be symmetric.
    """
    n_commits = len(commits)
    distances = [[0] * n_commits for _ in range(n_commits)]
    handlers = []
    for i in range(n_commits):
        for j in range(i + 1, n_commits):
            proc = subprocess.Popen(
                ["git", "rev-list", "--count", f"{commits[i]}...{commits[j]}"],
                cwd=config.git_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            handlers.append((i, j, proc))

    for i, j, proc in handlers:
        stdout, stderr = proc.communicate()
        assert proc.returncode == 0, "Failed to calculate the distance."
        distances[i][j] = distances[j][i] = int(stdout.decode("utf-8").strip())

    return distances


def pairwise_shared_merge_base_distances(
    config: EvoGitConfig, commits: list[str]
) -> list[list[int]]:
    """Calculate the pairwise distances from the merge bases to the initial commits."""
    n_commits = len(commits)
    distances = [[0] * n_commits for _ in range(n_commits)]
    merge_bases_handlers = []
    rev_count_handlers = []
    for i in range(n_commits):
        for j in range(i + 1, n_commits):
            proc = subprocess.Popen(
                ["git", "merge-base", "--all", commits[i], commits[j]],
                cwd=config.git_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            merge_bases_handlers.append((i, j, proc))

    merge_bases = []
    for i, j, proc in merge_bases_handlers:
        stdout, stderr = proc.communicate()
        assert proc.returncode == 0, "Failed to calculate the merge base."
        merge_base = stdout.decode("utf-8").strip().split()
        merge_bases.append((i, j, merge_base))

    for i, j, merge_base in merge_bases:
        proc = subprocess.Popen(
            ["git", "rev-list", "--count"] + merge_base,
            cwd=config.git_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        rev_count_handlers.append((i, j, proc))

    for i, j, proc in rev_count_handlers:
        stdout, stderr = proc.communicate()
        assert proc.returncode == 0, "Failed to calculate the distance."
        distances[i][j] = distances[j][i] = int(stdout.decode("utf-8").strip())

    return distances


def push_to_remote(
    config: EvoGitConfig, branches: list[str], async_push: bool = True
) -> subprocess.Popen | None:
    """Push the branch to the remote repository along side with the notes."""
    cmd = ["git", "push", "-q", "-f", "--atomic", "origin"]
    cmd += branches
    if async_push:
        # spawn the push process and return immediately
        # don't wait for the push to finish
        return subprocess.Popen(
            cmd,
            cwd=config.git_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            cmd,
            cwd=config.git_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def push_notes_to_remote(
    config: EvoGitConfig, async_push: bool = True
) -> subprocess.Popen | None:
    """Push the notes to the remote repository."""
    remote_notes_namespace = f"refs/notes/{config.hostname}-commits"
    cmd = [
        "git",
        "push",
        "-q",
        "origin",
        f"refs/notes/commits:{remote_notes_namespace}",
    ]
    if async_push:
        # spawn the push process and return immediately
        # don't wait for the push to finish
        return subprocess.Popen(
            cmd,
            cwd=config.git_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            cmd,
            cwd=config.git_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def fetch_from_remote(
    config: EvoGitConfig, prune=True, async_fetch: bool = True
) -> subprocess.Popen | None:
    """Fetch the notes from the remote repository."""
    cmd = ["git", "fetch", "-q"]
    if prune:
        cmd.append("--prune")

    cmd.append("origin")

    if async_fetch:
        return subprocess.Popen(
            cmd,
            cwd=config.git_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            cmd,
            cwd=config.git_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def list_remote_notes_namespaces(config: EvoGitConfig) -> list[str]:
    """List all the remote notes namespaces."""
    refs = subprocess.run(
        ["git", "notes", "get-ref"],
        check=True,
        cwd=config.git_dir,
        capture_output=True,
    ).stdout.decode("utf-8")
    refs = refs.split("\n")
    # remove the empty string and the default notes namespace
    notes_namespaces = [ref[11:] for ref in refs if ref and ref != "refs/notes/commits"]
    return notes_namespaces


def merge_notes(config: EvoGitConfig) -> None:
    """Merge the notes from the remote repository."""
    remote_notes_namespaces = list_remote_notes_namespaces(config)
    for remote_notes_namespace in remote_notes_namespaces:
        subprocess.run(
            ["git", "notes", "merge", "--strategy", "ours", remote_notes_namespace],
            cwd=config.git_dir,
        )


def fetch_notes_from_remote(
    config: EvoGitConfig, async_fetch: bool = True
) -> subprocess.Popen | None:
    """Fetch the notes from the remote repository."""
    cmd = ["git", "fetch", "-q", "origin", "refs/notes/*:refs/notes/*"]

    if async_fetch:
        return subprocess.Popen(
            cmd,
            cwd=config.git_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            cmd,
            cwd=config.git_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def diff_view(
    config: EvoGitConfig, commit1: str, commit2: str, context_lines: int = 8
) -> None:
    """View the diff between two commits."""
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--no-color",
            f"--unified={str(context_lines)}",
            commit1,
            commit2,
        ],
        cwd=config.git_dir,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    return diff


def has_staged_changes(config: EvoGitConfig, worktree: str) -> bool:
    """Check if there are staged changes in the specified worktree."""
    changes = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=worktree,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    # if the output is empty, there are no staged changes
    if changes.strip() == "":
        return False
    else:
        return True


def list_files(
    config: EvoGitConfig, commit: str, return_format: str = "string"
) -> list[str]:
    """List all the files in the specified commit."""
    files = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit],
        cwd=config.git_dir,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    if return_format == "string":
        return files
    elif return_format == "list":
        files = files.split("\n")
        files = [file.strip() for file in files if file != ""]
        return files
    else:
        raise ValueError("return_format must be either 'string' or 'list'.")


def prune(config: EvoGitConfig) -> None:
    """Run git prune."""
    gc_log = os.path.join(config.git_dir, ".git", "gc.log")
    if os.path.exists(gc_log):
        try:
            os.remove(gc_log)
        except FileNotFoundError:
            pass

    subprocess.run(
        ["git", "gc"],
        cwd=config.git_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "notes", "prune"],
        cwd=config.git_dir,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )
