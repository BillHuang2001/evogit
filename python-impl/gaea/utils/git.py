"""
The utility functions for git operations.
"""

import subprocess
import os
import shutil
import tempfile
import uuid
from gaea.config import GAEAConfig
from typing import Optional
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


def delete_remote_branches(config: GAEAConfig) -> None:
    fetch_from_remote(config)
    remote_branches = list_branches(config, list_remote=True)
    print(f"Delete the following remote branches: {remote_branches}")
    # delete all remote branches
    # delete branch cannot be done in parallel
    # thus run it sequentially
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


def init_git_repo(config: GAEAConfig) -> None:
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
    shutil.copy(config.seed_file, os.path.join(git_dir, config.filename))
    subprocess.run(["git", "add", "."], cwd=git_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=git_dir, check=True)

    if config.remote_repo is not None:
        subprocess.run(
            ["git", "remote", "add", "origin", config.remote_repo],
            cwd=git_dir,
            check=True,
        )
        subprocess.run(
            ["git", "push", "-f", "origin", "master"], cwd=git_dir, check=True
        )

        delete_remote_branches(config)


def clone_git_repo(config: GAEAConfig) -> None:
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


def get_commit_by_branch(config: GAEAConfig, branch: str) -> str:
    return (
        subprocess.run(
            ["git", "rev-parse", branch],
            cwd=config.git_dir,
            check=True,
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .strip()
    )


def get_commit_by_tag(config: GAEAConfig, tag: str) -> str:
    return get_commit_by_branch(config, f"tags/{tag}")


def read_head_commit(config: GAEAConfig) -> str:
    """Read the commit id of the current HEAD."""
    return get_commit_by_branch(config, "HEAD")


def list_branches(
    config: GAEAConfig, list_remote: bool = False, ea_only: bool = True
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


def list_tags(config: GAEAConfig) -> list[str]:
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


def delete_tags(config: GAEAConfig, tags: list[str]) -> None:
    """Delete the specified tags."""
    if tags:
        subprocess.run(
            ["git", "tag", "-d"] + tags,
            cwd=config.git_dir,
            check=True,
        )


def add_note(
    config: GAEAConfig, commit: str, note: str, overwrite: bool = False
) -> None:
    """Add a note to the current commit. If overwrite is True, force overwrite the existing note."""

    cmd = ["git", "notes", "add", "-m", note]
    if overwrite:
        cmd.append("-f")

    cmd.append(commit)

    subprocess.run(cmd, cwd=config.git_dir, check=True)


def read_note(config: GAEAConfig, commit: Optional[str]) -> Optional[str]:
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


def update_file(
    config: GAEAConfig, commit: str, new_content: str, commit_message: str
) -> None:
    """Update the content of the file in the specified commit_id and commit the updated file."""
    checkout(config, commit)
    with open(os.path.join(config.git_dir, config.filename), "r+") as f:
        current_content = f.read()
        # if the content is the same, do nothing
        if current_content == new_content:
            return

        f.seek(0)
        f.write(new_content)
        f.truncate()

    subprocess.run(["git", "add", config.filename], cwd=config.git_dir, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", commit_message], cwd=config.git_dir, check=True
    )


def read_file(config: GAEAConfig, commit: str) -> str:
    """Read the content of the file in the specified commit."""
    return subprocess.run(
        ["git", "show", f"{commit}:{config.filename}"],
        cwd=config.git_dir,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")


def batch_read_files(config: GAEAConfig, commits: list[str]) -> list[str]:
    return [read_file(config, commit) for commit in commits]


def has_conflict(config: GAEAConfig) -> bool:
    """Return True if the current working directory has conflicts. Otherwise, return False."""
    status = subprocess.run(
        ["git", "status"], cwd=config.git_dir, capture_output=True, check=True
    ).stdout.decode("utf-8")

    if "Unmerged paths" in status or "rebasing" in status or "merging" in status:
        return True
    else:
        return False


def count_conflicts(config: GAEAConfig) -> int:
    """Count the number of conflicts in the current working directory."""
    with open(os.path.join(config.git_dir, config.filename), "r") as f:
        content = f.read()

    return len(git_conflict_pattern.findall(content))


def checkout(config: GAEAConfig, commit: str) -> None:
    """Checkout the specified commit."""
    # -q is quiet, --detach is used to checkout the commit in detached HEAD mode
    subprocess.run(
        ["git", "checkout", "-q", "--detach", commit], cwd=config.git_dir, check=True
    )


def add_temp_worktree(config: GAEAConfig, branch: str) -> str:
    """checkout the branch in a new worktree and return the path of the worktree."""
    # make a temporary directory if not exists
    worktree_dir = os.path.join(config.git_dir, ".gaea_evaluate")
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
    )
    return worktree


def remove_temp_worktree(config: GAEAConfig, worktree: str) -> None:
    """Remove the worktree of the branch."""
    subprocess.run(
        ["git", "worktree", "remove", "-f", worktree],
        cwd=config.git_dir,
        check=True,
    )


def cleanup_temp_worktrees(config: GAEAConfig) -> None:
    """Remove all the worktrees in the .gaea_evaluate directory."""
    worktree_dir = os.path.join(config.git_dir, ".gaea_evaluate")
    if os.path.exists(worktree_dir):
        shutil.rmtree(worktree_dir)

    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=config.git_dir,
        check=True,
    )


def merge_branches(config: GAEAConfig, commit: str) -> None:
    """merge the commit specified by the commit_id to the current branch."""
    # don't check the return code because the merge may fail
    subprocess.run(["git", "merge", "-q", commit, "--no-edit"], cwd=config.git_dir)


def rebase_branches(config: GAEAConfig, commit: str) -> None:
    """rebase the current branch on the commit specified by the commit_id."""
    # don't check the return code because the rebase may fail
    # GIT_EDITOR=true is used to disable the interactive editor
    # and accept the default commit message
    subprocess.run(
        ["git", "rebase", "-q", commit],
        cwd=config.git_dir,
        env=os.environb | {"GIT_EDITOR": "true"},
    )


def continue_merge(config: GAEAConfig) -> None:
    """Continue the merge process."""
    subprocess.run(
        ["git", "merge", "--continue"],
        cwd=config.git_dir,
        env=os.environb | {"GIT_EDITOR": "true"},
        check=True,
    )


def continue_rebase(config: GAEAConfig) -> None:
    """Continue the rebase process."""
    # GIT_EDITOR=true is used to disable the interactive editor
    # and accept the default commit message
    subprocess.run(
        ["git", "rebase", "--continue"],
        cwd=config.git_dir,
        env=os.environb | {"GIT_EDITOR": "true"},
    )


def handle_conflict(config: GAEAConfig, strategy: list[bool]) -> None:
    """Handle the conflict by accept ours or theirs.
    The strategy is a list of bool values, where True means accepting ours, and False means accepting theirs.
    Write back the result to the file and git add.
    """
    with open(os.path.join(config.git_dir, config.filename), "r") as f:
        content = f.read()

    iterator = iter(strategy)

    def handle_one_conflict(match):
        accept_ours = next(iterator)
        if accept_ours:
            return match.group(1)
        else:
            return match.group(2)

    result = git_conflict_pattern.sub(handle_one_conflict, content)
    with open(os.path.join(config.git_dir, config.filename), "w") as f:
        f.write(result)

    subprocess.run(["git", "add", config.filename], cwd=config.git_dir, check=True)


def branches_track_commits(
    config: GAEAConfig, branch_names: list[str], commits: list[str]
) -> None:
    """Create branches that track the specified commits."""
    for branch_name, commit in zip(branch_names, commits):
        subprocess.run(
            ["git", "branch", "-f", branch_name, commit], cwd=config.git_dir, check=True
        )


def push_to_remote(
    config: GAEAConfig, branches: list[str], async_push: str = True
) -> None:
    """Push the branch to the remote repository along side with the notes."""
    cmd = ["git", "push", "-q", "-f", "--atomic", "origin"]
    cmd += branches
    if async_push:
        # spawn the push process and return immediately
        # don't wait for the push to finish
        subprocess.Popen(
            cmd,
            cwd=config.git_dir,
        )
    else:
        subprocess.run(
            cmd,
            cwd=config.git_dir,
            check=True,
        )


def fetch_from_remote(config: GAEAConfig, prune=True) -> None:
    """Fetch the notes from the remote repository."""
    cmd = ["git", "fetch", "-q"]
    if prune:
        cmd.append("--prune")

    cmd.append("origin")

    subprocess.run(
        cmd,
        cwd=config.git_dir,
        check=True,
    )
