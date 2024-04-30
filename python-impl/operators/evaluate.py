import subprocess
import os
from config import GAEAConfig
import utils
import json


def evalute(config: GAEAConfig, tag: str, cmd: list[str], timeout=60, sandbox=True):
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
    """
    if sandbox:
        # run the command in a bubblewrap sandbox
        cmd = ["bubblewrap_run.sh"] + cmd

    utils.git.checkout(config, tag)

    try:
        completed_proc = subprocess.run(
            cmd,
            capture_output=True,
            cwd=config.git_dir,
            timeout=timeout,
        )
        stdout = completed_proc.stdout.decode("utf-8")
        result = {
            "stdout": stdout,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        result = {
            "stdout": "",
            "timeout": True,
        }
    
    note = json.dump(result)
    utils.git.add_note(config, note)
    return result
