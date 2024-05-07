from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GAEAConfig:
    llm_name: str
    device_map: str
    git_dir: str
    seed_file: str
    filename: str
    merge_prob: float
    accept_ours_prob: float
    sign_tag: bool
    reevaluate: bool
    remote_repo: Optional[str]
    hostname: Optional[str]

    def __hash__(self):
        return id(self)
