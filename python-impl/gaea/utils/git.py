"""
The utility functions for git operations.
"""

import subprocess
import os
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


def init_git_repo(config: GAEAConfig):
    git_dir = config.git_dir
    if os.path.exists(git_dir):
        print(f"{git_dir} already exists! Do you want to remove it? (y/n)")
        if input() == "y":
            print(f"Removing {git_dir}...")
            os.system(f"rm -rf {git_dir}")
        else:
            print("Abort!")
            exit()

    os.system(f"mkdir -p {git_dir}")
    subprocess.run(["git", "init"], cwd=git_dir, check=True)
    subprocess.run(
        ["cp", config.seed_file, os.path.join(git_dir, config.filename)], check=True
    )
    subprocess.run(["git", "add", "."], cwd=git_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=git_dir, check=True)


def read_head_commit(config: GAEAConfig):
    """Read the commit id of the current HEAD."""
    return (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.git_dir,
            check=True,
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .strip()
    )


def add_note(config: GAEAConfig, note: str, overwrite: bool = False):
    """Add a note to the current commit. If overwrite is True, force overwrite the existing note."""

    cmd = ["git", "notes", "add", "-m", note]
    if overwrite:
        cmd.append("-f")

    subprocess.run(cmd, cwd=config.git_dir, check=True)


def read_note(config: GAEAConfig, commit: Optional[str]):
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


def update_file(config: GAEAConfig, commit: str, new_content: str, commit_message: str):
    """Update the content of the file in the specified commit_id and commit the updated file."""
    subprocess.run(["git", "checkout", commit], cwd=config.git_dir, check=True)
    with open(os.path.join(config.git_dir, config.filename), "w") as f:
        f.write(new_content)
    subprocess.run(["git", "add", config.filename], cwd=config.git_dir, check=True)
    subprocess.run(
        ["git", "commit", "-m", commit_message], cwd=config.git_dir, check=True
    )


def read_file(config: GAEAConfig, commit: str):
    """Read the content of the file in the specified commit."""
    return subprocess.run(
        ["git", "show", f"{commit}:{config.filename}"],
        cwd=config.git_dir,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")


def batch_read_files(config: GAEAConfig, commits: list[str]):
    return [read_file(config, commit) for commit in commits]


def has_conflict(config: GAEAConfig):
    """Return True if the current working directory has conflicts. Otherwise, return False."""
    status = subprocess.run(
        ["git", "status"], cwd=config.git_dir, capture_output=True, check=True
    ).stdout.decode("utf-8")

    if "Unmerged paths" in status:
        return True
    else:
        return False


def count_conflicts(config: GAEAConfig):
    """Count the number of conflicts in the current working directory."""
    with open(os.path.join(config.git_dir, config.filename), "r") as f:
        content = f.read()

    return len(git_conflict_pattern.findall(content))


def checkout(config: GAEAConfig, commit: str):
    """Checkout the specified commit."""
    subprocess.run(
        ["git", "checkout", "--detach", commit], cwd=config.git_dir, check=True
    )


def merge_branches(config: GAEAConfig, commit: str):
    """merge the commit specified by the commit_id to the current branch."""
    subprocess.run(
        ["git", "merge", commit, "--no-edit"], cwd=config.git_dir, check=True
    )


def rebase_branches(config: GAEAConfig, commit: str):
    """rebase the current branch on the commit specified by the commit_id."""
    subprocess.run(
        ["git", "rebase", commit],
        cwd=config.git_dir,
        env={"GIT_EDITOR": "true"},
        check=True,
    )


def continue_merge(config: GAEAConfig):
    """Continue the merge process."""
    subprocess.run(["git", "merge", "--continue"], cwd=config.git_dir, check=True)


def continue_rebase(config: GAEAConfig):
    """Continue the rebase process."""
    subprocess.run(["git", "rebase", "--continue"], cwd=config.git_dir, check=True)


def handle_conflict(config: GAEAConfig, strategy: list[bool]):
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
):
    """Create branches that track the specified commits."""
    for branch_name, commit in zip(branch_names, commits):
        subprocess.run(
            ["git", "branch", "-f", branch_name, commit], cwd=config.git_dir, check=True
        )


def push_branch_to_remote(config: GAEAConfig, branch_name: str, remote: str):
    """Push the branch to the remote repository along side with the notes."""
    checkout(config, branch_name)
    subprocess.run(
        ["git", "push", "-f", remote, branch_name], cwd=config.git_dir, check=True
    )
    subprocess.run(
        ["git", "push", remote, f"{branch_name}:refs/notes/commits"],
        cwd=config.git_dir,
        check=True,
    )
