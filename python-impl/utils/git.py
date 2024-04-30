"""
The utility functions for git operations.
"""

import subprocess
import os
import uuid
from config import GAEAConfig
import re


git_conflict_pattern = re.compile(
    r"<<<<<<<.*?\n(.*?)=======.*?\n(.*?)>>>>>>>.*?\n", re.DOTALL
)


def alloc_new_tag_name():
    return uuid.uuid1().hex


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
    subprocess.run(["git", "init"], cwd=git_dir)
    subprocess.run(["cp", config.seed_file, os.path.join(git_dir, config.filename)])
    subprocess.run(["git", "add", "."], cwd=git_dir)
    subprocess.run(["git", "commit", "-m", "init"], cwd=git_dir)


def tag(config: GAEAConfig, tag):
    if config.sign_tag:
        subprocess.run(["git", "tag", "-s", tag], cwd=config.git_dir)
    else:
        subprocess.run(["git", "tag", tag], cwd=config.git_dir)


def delete_tag(config: GAEAConfig, tag: str):
    subprocess.run(["git", "tag", "-d", tag], cwd=config.git_dir)


def add_note(config: GAEAConfig, note: str):
    subprocess.run(["git", "notes", "add", "-m", note], cwd=config.git_dir)


def update_file(config: GAEAConfig, tag: str, new_content: str, commit_message: str):
    """Update the content of the file in the specified tag and commit it."""
    subprocess.run(["git", "checkout", tag], cwd=config.git_dir)
    with open(os.path.join(config.git_dir, config.filename), "w") as f:
        f.write(new_content)
    subprocess.run(["git", "add", config.filename], cwd=config.git_dir)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=config.git_dir)


def read_file(config: GAEAConfig, tag: str):
    """Read the content of the file in the specified tag."""
    subprocess.run(["git", "show", f"{tag}:{config.filename}"], cwd=config.git_dir)


def batch_read_files(config: GAEAConfig, tags: list[str]):
    return [read_file(config, tag) for tag in tags]


def has_conflict(config: GAEAConfig):
    """Return True if the current working directory has conflicts. Otherwise, return False."""
    status = subprocess.run(
        ["git", "status"], cwd=config.git_dir, capture_output=True
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


def checkout(config: GAEAConfig, tag):
    """Checkout the specified tag."""
    subprocess.run(["git", "checkout", tag], cwd=config.git_dir)


def merge_branches(config: GAEAConfig, tag):
    """merge the commit specified by the tag to the current branch."""
    subprocess.run(["git", "merge", tag, "--no-edit"], cwd=config.git_dir)


def continue_merge(config: GAEAConfig):
    """Continue the merge process."""
    subprocess.run(["git", "merge", "--continue"], cwd=config.git_dir)


def continue_rebase(config: GAEAConfig):
    """Continue the rebase process."""
    subprocess.run(["git", "rebase", "--continue"], cwd=config.git_dir)


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

    subprocess.run(["git", "add", config.filename], cwd=config.git_dir)
