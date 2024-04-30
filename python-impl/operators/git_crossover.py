import numpy as np
from config import GAEAConfig
import utils


def git_crossover(config: GAEAConfig, seed: int, tag1: str, tag2: str):
    """crossover between tag1 and tag2"""
    rng = np.random.default_rng(seed)
    use_merge = rng.choice([True, False], p=[config.merge_prob, 1 - config.merge_prob])
    if use_merge:
        git_merge(config, rng, tag1, tag2)
    else:
        git_rebase(config, rng, tag1, tag2)


def git_merge(config: GAEAConfig, rng: np.random.Generator, tag1: str, tag2: str):
    utils.git.checkout(config, tag1)
    utils.git.merge_branches(config, tag2)

    if utils.git.has_conflict(config):
        count = utils.git.count_conflicts(config)
        strategy = rng.choice(
            [True, False],
            size=(count,),
            p=[config.accept_ours_prob, 1 - config.accept_ours_prob],
        )
        utils.git.handle_conflict(config, strategy)
        utils.git.continue_merge(config)
    
    new_tag = utils.git.alloc_new_tag_name()
    utils.git.tag(config, new_tag)

    return new_tag


def git_rebase(config: GAEAConfig, rng: np.random.Generator, tag1: str, tag2: str):
    utils.git.checkout(config, tag1)
    utils.git.rebase_branches(config, tag2)

    while utils.git.has_conflict(config):
        count = utils.git.count_conflicts(config)
        strategy = rng.choice(
            [True, False],
            size=(count,),
            p=[config.accept_ours_prob, 1 - config.accept_ours_prob],
        )
        utils.git.handle_conflict(config, strategy)
        utils.git.continue_rebase(config)

    new_tag = utils.git.alloc_new_tag_name()
    utils.git.tag(config, new_tag)

    return new_tag