from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GAEAConfig:
    git_user_name: str
    git_user_email: str
    llm_name: str
    device_map: str
    git_dir: str
    eval_command: list[str]
    seed_file: str
    filename: str
    merge_prob: float
    accept_ours_prob: float
    git_hash: str # sha1 or sha256
    reevaluate: bool
    api_key: Optional[str]
    http_req_params: dict
    remote_repo: Optional[str]
    hostname: Optional[str]

    def __hash__(self):
        return id(self)
