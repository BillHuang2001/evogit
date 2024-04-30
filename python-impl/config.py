from dataclasses import dataclass


@dataclass
class GAEAConfig:
    llm_names: list[str]
    git_dir: str
    seed_file: str
    filename: str
    merge_prob: float
    accept_ours_prob: float
    sign_tag: bool
