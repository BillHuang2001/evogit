from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class GAEAConfig:
    num_objectives: int
    git_user_name: str
    git_user_email: str
    push_every: int
    fetch_every: int
    migrate_every: int
    human_every: int
    migrate_count: int
    llm_name: str
    llm_backend: Any
    device_map: str
    git_dir: str
    eval_command: list[str]
    seed_file: str
    filename: str
    merge_prob: float
    accept_ours_prob: float
    git_hash: str # sha1 or sha256
    evaluate_workers: int
    reevaluate: bool
    enable_sandbox: bool
    timeout: int
    prompt_constructor: callable
    respond_extractor: callable
    remote_repo: Optional[str]
    hostname: Optional[str]

    def __hash__(self):
        return id(self)
