import json
import warnings
from functools import partial
from pathlib import Path
import logging

import jax
import jax.numpy as jnp
from evox import Monitor, Problem, jit_class
from jax.experimental import io_callback

from gaea import api


# commit_id is either sha1 - 160 bits or sha256 - 256 bits
# we use 20 bytes to represent the commit id using sha1
# and 32 bytes to represent the commit id using sha256


def array_to_hex(array):
    return array.tobytes().hex()


def hex_to_array(hex_string):
    return jnp.frombuffer(bytes.fromhex(hex_string), dtype=jnp.uint8)


HASH_BYTE_LENGTH = {
    "sha1": 20,
    "sha256": 32,
}


class BranchMonitor(Monitor):
    def __init__(
        self,
        config,
        population_name="population",
    ):
        super().__init__()
        self.population_name = population_name

        def update_branches(pop):
            pop = [array_to_hex(individual) for individual in pop]
            api.update_branches(config, pop)

        self.update_branches = update_branches

    def hooks(self):
        return ["post_step"]

    def post_step(self, state):
        population = getattr(state.get_child_state("algorithm"), self.population_name)
        io_callback(self.update_branches, None, population)

    def get_population_history(self):
        return self.population_history

    def get_fitness_history(self):
        return self.fitness_history


def gaea_git_crossover(config, seeds, parents):
    offspring = []
    for seed, commit1, commit2 in zip(seeds, parents[0], parents[1]):
        commit1 = array_to_hex(commit1)
        commit2 = array_to_hex(commit2)
        new_commit = api.git_crossover(config, seed, commit1, commit2)
        offspring.append(hex_to_array(new_commit))

    return jnp.stack(offspring)


@partial(jax.jit, static_argnums=(0, 1))
def git_crossover(config, type, key, parents):
    pop_size, dim = parents.shape
    num_pairs = pop_size // 2
    parents = parents.reshape(2, num_pairs, dim)
    if type == 2:
        num_pairs = pop_size
        parents = jnp.concatenate([parents, parents[::-1]], axis=1)
    seeds = jax.random.randint(key, (num_pairs,), 0, jnp.iinfo(jnp.int32).max)

    byte_length = HASH_BYTE_LENGTH[config.git_hash]
    return_type = jax.ShapeDtypeStruct((num_pairs, byte_length), jnp.uint8)
    return io_callback(partial(gaea_git_crossover, config), return_type, seeds, parents)


def gaea_llm_mutation(config, llm_backend, seeds, pop):
    commits = [array_to_hex(commit) for commit in pop]
    new_commits = api.llm_mutation(config, llm_backend, seeds, commits)
    offspring = [hex_to_array(new_commit) for new_commit in new_commits]

    return jnp.stack(offspring)


@partial(jax.jit, static_argnums=(0, 1))
def llm_mutation(config, llm_backend, key, pop):
    pop_size = pop.shape[0]
    seeds = jax.random.randint(key, (pop_size,), 0, jnp.iinfo(jnp.int32).max)

    byte_length = HASH_BYTE_LENGTH[config.git_hash]
    return_type = jax.ShapeDtypeStruct((pop_size, byte_length), jnp.uint8)
    return io_callback(
        partial(gaea_llm_mutation, config, llm_backend), return_type, seeds, pop
    )


@jit_class
class GitCrossover:
    def __init__(self, config, type=2):
        self.config = config
        self.type = type

    def __call__(self, key, parents):
        return git_crossover(self.config, type, key, parents)


@jit_class
class LLMMutation:
    def __init__(self, config):
        self.config = config
        self.llm_backend = api.prepare_llm_backend(config)

    def __call__(self, key, pop):
        return llm_mutation(self.config, self.llm_backend, key, pop)


def evaluate(config, pop):
    pop = [array_to_hex(individual) for individual in pop]
    logger = logging.getLogger("gaea")
    logger.info(pop)
    output = [
        api.evaluate(
            config,
            commit,
            config.eval_command,
        )
        for commit in pop
    ]
    if config.num_objectives == 1:
        illegal_value = jnp.inf
    else:
        illegal_value = [jnp.inf for _ in range(config.num_objectives)]
    fitness = []
    for o in output:
        if o["timeout"] == True:
            fit = illegal_value
        else:
            try:
                stdout = json.loads(o["stdout"])
                if stdout["status"] == "finished":
                    fit = stdout["fitness"]
                else:
                    fit = illegal_value
            except json.JSONDecodeError:
                fit = illegal_value
            except Exception as e:
                logger.error(f"Unknown error occurred: {e}")
                fit = illegal_value
        fitness.append(fit)
    return jnp.array(fitness)


@jit_class
class CodegenProblem(Problem):
    def __init__(self, config):
        self.config = config

    def evaluate(self, state, population):
        if self.config.num_objectives == 1:
            return_dtype = jax.ShapeDtypeStruct((population.shape[0],), jnp.float32)
        else:
            return_dtype = jax.ShapeDtypeStruct(
                (population.shape[0], self.config.num_objectives), jnp.float32
            )

        return (
            io_callback(partial(evaluate, self.config), return_dtype, population),
            state,
        )


def init_population(config, pop_size):
    pop = api.get_initial_branches(config, pop_size)
    pop = [hex_to_array(commit) for commit in pop]
    return jnp.stack(pop)
