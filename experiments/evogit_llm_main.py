import sys
import logging
from datetime import datetime
import re
import os
import torch
from typing import NamedTuple
from evox.workflows import StdWorkflow

sys.path.append("./python-impl")
sys.path.append("./experiments")

from evogit_algorithm import EvoGitAlgo

torch.set_default_device("cpu")

from evogit_llm_prompt import (
    basic_info,
    mutation_template,
    diff_template,
)
from evogit.config import EvoGitConfig
from evox_extension import EvoGitProblem, api, update_branches, array_to_hex, hex_to_array, git_update
import evogit


host_id = sys.argv[1]
llm_name = sys.argv[2]
remote_repo = "git@github.com-evogit-llm:BillHuang2001/evogit_llm.git"
stages_dir = "./design_doc/evogit_llm_stages"
base_name = "stage"
STAGE = 0
print("host_id", host_id)
print("llm_name", llm_name)

username = "bchuang"

logger = logging.getLogger("evogit")
logger.propagate = False  # Disable the default printing behavior
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
f_handler = logging.FileHandler(f"log/evogit_{timestamp}.log")
f_handler.setLevel("DEBUG")
s_handler = logging.StreamHandler(sys.stdout)
s_handler.setLevel("WARNING")
logger.setLevel("DEBUG")
logger.addHandler(f_handler)
logger.addHandler(s_handler)

logger.warning(f"host id: {host_id}")
logger.warning(f"llm name: {llm_name}")

with open("./secrets/azure_token", "r") as f:
    api_token = f.read().strip()

# check if the directory exists
if not os.path.exists(stages_dir):
    raise ValueError(f"Directory {stages_dir} does not exist")


def prompt_constructor(file_list, filename, prompt_code, lint_output):
    global STAGE
    stage_file = os.path.join(stages_dir, f"{base_name}_{STAGE}.md")
    with open(stage_file, "r") as f:
        current_task = f.read()

    return mutation_template.format(
        structure=file_list,
        filename=filename,
        code=prompt_code,
        lint=lint_output,
        current_task=current_task,
    )


code_extract_pattern = re.compile(r"```.*?\n(.*?)```", re.DOTALL)
filename_pattern = re.compile(r"^`([^`]+)`", re.MULTILINE)


class ResponseContent(NamedTuple):
    code: str
    filename: str
    new_file_content: str
    commit_message: str


def respond_extractor(response: str) -> ResponseContent:
    try:
        code_blocks = code_extract_pattern.findall(response)
        filename_match = filename_pattern.search(response)
        assert len(code_blocks) == 3, f"Expected 3 code blocks, got {len(code_blocks)}"

        # Extract fields with safe fallbacks
        code = code_blocks[0].strip() + "\n" if len(code_blocks) > 0 else ""
        new_file_content = code_blocks[1].strip() + "\n" if len(code_blocks) > 1 else ""
        commit_message = (
            code_blocks[2].strip() if len(code_blocks) > 2 else "LLM code update"
        )
        commit_message = commit_message[:256]  # Truncate to 256 characters

        filename = filename_match.group(1).strip() if filename_match else "None"
        return ResponseContent(
            code=code,
            filename=filename,
            new_file_content=new_file_content,
            commit_message=commit_message,
        )

    except Exception as e:
        logger.warning(
            f"Error in response extraction, original response: {response}; error: {e}."
        )
        return ResponseContent(
            code="",
            filename="None",
            new_file_content="",
            commit_message="LLM code update",
        )


def diff_prompt_constructor(file_list, diff, prev_note, new_note):
    global STAGE
    stage_file = os.path.join(stages_dir, f"{base_name}_{STAGE}.md")
    with open(stage_file, "r") as f:
        current_task = f.read()

    return diff_template.format(
        structure=file_list,
        diff=diff,
        prev_lint=prev_note,
        new_lint=new_note,
        current_task=current_task,
    )


http_req_params = {
    "timeout": 600,
}
host = "127.0.0.1"
azure_endpoint = "https://polaris-azure-openai-api.openai.azure.com/"
model = llm_name.strip()

llm_backend = evogit.utils.llm.TGIBackend(
    endpoint=azure_endpoint,
    api_key=api_token,
    model=model,
    num_workers=1,
    top_p=0.7,
)

config = EvoGitConfig(
    num_objectives=0,
    git_user_name="Bill Huang",
    git_user_email="bill.huang2001@gmail.com",
    push_every=1,
    fetch_every=0,
    migrate_every=1,
    human_every=0,
    migrate_count=0,
    llm_name=llm_name,
    llm_backend=llm_backend,
    device_map="auto",
    git_dir=f"/tmp/evogit/evogit_llm_{llm_name}_{host_id}",
    eval_command=None,
    seed_file=None,
    filename=None,
    merge_prob=1,
    accept_ours_prob=0.5,
    git_hash="sha1",
    evaluate_workers=120,
    reevaluate=False,
    enable_sandbox=False,
    timeout=10,
    prompt_constructor=prompt_constructor,
    respond_extractor=respond_extractor,
    diff_prompt_constructor=diff_prompt_constructor,
    fixup_prompt_constructor=None,
    max_merge_retry=512,
    clean_start=True,
    project_type="python",
    remote_repo=remote_repo,
    hostname="host" + host_id,
    merge_driver=None,
)


api.init_repo(config, "remote", force_create=True)
n_iter = 120
human_feedback_every = 20

algorithm = EvoGitAlgo(
    config,
    pop_size=16,
    crossover_every=3,
)
problem = EvoGitProblem(config)
workflow = StdWorkflow(algorithm, problem)
population_history = []


try:
    for i in range(n_iter):
        logger.warning(f"Iteration {i}    Stage {STAGE}")
        if i == 0:
            workflow.init_step()
        else:
            workflow.step()
        population = workflow.algorithm.pop
        update_branches(config, population)
        git_update(config, i)
        population = [array_to_hex(individual) for individual in population]
        population_history.append(population)
        # save the data every 10 iterations

        if (i + 1) % human_feedback_every == 0:
            STAGE += 1
            logger.warning(f"Human feedback phase, stage {STAGE}")
            # pause the program and wait for human feedback
            # print the current population
            print("Current population:")
            print(population)
            # pause, and ask "y/n"
            manual_selection = input("Do you wish to select a commit id? (y/n): ")
            while manual_selection not in ["y", "n"]:
                manual_selection = input("Do you wish to select a commit id? (y/n): ")
            if manual_selection == "y":
                # ask for the commit id
                commit_id = input("Please enter the commit id: ")
                # check if the commit id is in the population
                while commit_id not in population:
                    print("Commit id not in the population, please try again.")
                    commit_id = input("Please enter the commit id: ")

                commit_id_arr = hex_to_array(commit_id)
                # set the selected commit id
                workflow.algorithm.pop = torch.broadcast_to(commit_id_arr, workflow.algorithm.pop.shape)

            feedback = input("Do you want to continue? (y/n): ")
            while feedback not in ["y", "n"]:
                feedback = input("Do you want to continue? (y/n): ")
            if feedback == "n":
                logger.warning("Stop")
                break
            elif feedback == "y":
                logger.warning("Continue")
except KeyboardInterrupt:
    pass
finally:
    print("Exit")
