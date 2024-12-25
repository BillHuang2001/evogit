from concurrent.futures import ThreadPoolExecutor
import json
from functools import partial
import logging

import jax
import jax.numpy as jnp
from evox import Monitor, Problem, jit_class
from jax.experimental import io_callback

from phylox import api


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
        sync_to_remote=True,
    ):
        super().__init__()
        self.config = config
        self.generation = 0
        self.population_name = population_name
        self.sync_to_remote = sync_to_remote

        def update_branches(pop):
            pop = [array_to_hex(individual) for individual in pop]
            api.update_branches(config, pop)
            api.prune_commits(self.config)

        self.update_branches = update_branches

    def hooks(self):
        return ["post_step"]

    def clear_history(self):
        return

    def post_step(self, state, workflow_state):
        population = getattr(
            workflow_state.get_child_state("algorithm"), self.population_name
        )
        state = state.register_callback(self.update_branches, population)
        if self.sync_to_remote:
            state = state.register_callback(self.git_update)
        return state

    def git_update(self):
        handlers = []
        if self.generation % self.config.fetch_every == 0:
            handlers.extend(api.fetch_remote(self.config))
        if self.generation % self.config.push_every == 0:
            handlers.extend(api.push_local_branches(self.config))

        for proc in handlers:
            proc.wait()

        self.generation += 1

    def get_population_history(self):
        return self.population_history

    def get_fitness_history(self):
        return self.fitness_history


def phylox_git_crossover(config, seeds, parents):
    offspring = []
    for seed, commit1, commit2 in zip(seeds, parents[0], parents[1]):
        commit1 = array_to_hex(commit1)
        commit2 = array_to_hex(commit2)
        new_commit = api.git_crossover(config, seed.item(), commit1, commit2)
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
    return io_callback(
        partial(phylox_git_crossover, config), return_type, seeds, parents
    )


def phylox_llm_mutation(config, llm_backend, seeds, pop):
    commits = [array_to_hex(commit) for commit in pop]
    seeds = seeds.tolist()
    new_commits = api.llm_mutation(config, llm_backend, seeds, commits)
    offspring = [hex_to_array(new_commit) for new_commit in new_commits]

    return jnp.stack(offspring)


def phylox_llm_crossover(config, llm_backend, seeds, pop):
    commits = [array_to_hex(commit) for commit in pop]
    seeds = seeds.tolist()
    new_commits = api.llm_crossover(config, llm_backend, seeds, commits)
    offspring = [hex_to_array(new_commit) for new_commit in new_commits]

    return jnp.stack(offspring)


@partial(jax.jit, static_argnums=(0, 1))
def llm_mutation(config, llm_backend, key, pop):
    pop_size = pop.shape[0]
    seeds = jax.random.randint(key, (pop_size,), 0, jnp.iinfo(jnp.int32).max)

    byte_length = HASH_BYTE_LENGTH[config.git_hash]
    return_type = jax.ShapeDtypeStruct((pop_size, byte_length), jnp.uint8)
    return io_callback(
        partial(phylox_llm_mutation, config, llm_backend), return_type, seeds, pop
    )


@partial(jax.jit, static_argnums=(0, 1))
def llm_crossover(config, llm_backend, key, pop):
    pop_size = pop.shape[0]
    seeds = jax.random.randint(key, (pop_size // 2,), 0, jnp.iinfo(jnp.int32).max)

    byte_length = HASH_BYTE_LENGTH[config.git_hash]
    return_type = jax.ShapeDtypeStruct((pop_size // 2, byte_length), jnp.uint8)
    return io_callback(
        partial(phylox_llm_crossover, config, llm_backend), return_type, seeds, pop
    )


@jit_class
class GitCrossover:
    def __init__(self, config, type=1):
        self.config = config
        self.type = type

    def __call__(self, key, parents):
        return git_crossover(self.config, self.type, key, parents)


@jit_class
class LLMMutation:
    def __init__(self, config):
        self.config = config
        self.llm_backend = config.llm_backend

    def __call__(self, key, pop):
        return llm_mutation(self.config, self.llm_backend, key, pop)


@jit_class
class LLMCrossover:
    def __init__(self, config):
        self.config = config
        self.llm_backend = config.llm_backend

    def __call__(self, key, pop):
        return llm_crossover(self.config, self.llm_backend, key, pop)


def evaluate(config, pool, pop):
    pop = [array_to_hex(individual) for individual in pop]
    logger = logging.getLogger("phylox")
    logger.info(pop)

    # 1. prepare worktrees  2. evaluate  3. update notes  4. cleanup worktrees
    unique_pop = list(set(pop))  # deduplicate
    worktrees = api.prepare_temp_worktrees(config, unique_pop)
    outputs = list(pool.map(partial(api.evaluate_code, config), unique_pop, worktrees))
    api.update_notes(config, unique_pop, outputs)
    api.cleanup_temp_worktrees(config)

    if config.num_objectives == 1:
        illegal_value = jnp.inf
    else:
        illegal_value = [jnp.inf for _ in range(config.num_objectives)]
    commit_to_fitness = {}
    for commit_id, output in zip(unique_pop, outputs):
        fit = api.decode_result(output, illegal_value)
        commit_to_fitness[commit_id] = fit
    fitness = [commit_to_fitness[commit_id] for commit_id in pop]
    return jnp.array(fitness).astype(jnp.float32)


@jit_class
class CodegenProblem(Problem):
    def __init__(self, config):
        self.config = config
        # ThreadPoolExecutor can achieve parallelism here
        # since the evaluate function will spawn new processes through subprocess.run
        self.pool = ThreadPoolExecutor(config.evaluate_workers)

    def evaluate(self, state, population):
        if self.config.num_objectives == 1:
            return_dtype = jax.ShapeDtypeStruct((population.shape[0],), jnp.float32)
        else:
            return_dtype = jax.ShapeDtypeStruct(
                (population.shape[0], self.config.num_objectives), jnp.float32
            )

        return (
            io_callback(
                partial(evaluate, self.config, self.pool), return_dtype, population
            ),
            state,
        )


def init_population(config, pop_size):
    pop = api.get_initial_branches(config, pop_size)
    pop = [hex_to_array(commit) for commit in pop]
    return jnp.stack(pop)


class MigrateHelper:
    def __init__(self, config):
        self.config = config
        self.generation = 0
        self.logger = logging.getLogger("phylox")

    def migrate_from_human(self):
        byte_length = HASH_BYTE_LENGTH[self.config.git_hash]
        return_type = (
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((1, byte_length), jnp.uint8),
            jax.ShapeDtypeStruct((), jnp.float32),
        )
        return io_callback(self._migrate_from_human, return_type)

    def migrate_from_other_hosts(self):
        byte_length = HASH_BYTE_LENGTH[self.config.git_hash]
        return_type = (
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((self.config.migrate_count, byte_length), jnp.uint8),
            jax.ShapeDtypeStruct((self.config.migrate_count,), jnp.float32),
        )
        return io_callback(self._migrate_from_other_hosts, return_type)

    def _migrate_from_human(self):
        if self.generation % self.config.human_every == 0:
            commit, fitness = api.migrate_from_human_tags(self.config, 1)

            if commit:
                self.logger.info(f"Found commit by human: {commit}")
                return True, jnp.array([hex_to_array(commit[0])]), fitness

        return False, jnp.empty(
            (1, HASH_BYTE_LENGTH[self.config.git_hash]), dtype=jnp.uint8
        )

    def _migrate_from_other_hosts(self):
        if self.generation % self.config.migrate_every == 0:
            commits, fitness = api.migrate_from_other_hosts(
                self.config, self.config.migrate_count
            )

            if len(commits) == self.config.migrate_count:
                self.logger.info(f"Migrating commits from other hosts: {commits}")
                return (
                    True,
                    jnp.stack([hex_to_array(commit) for commit in commits]),
                    fitness,
                )

        self.logger.info("No commits found from other hosts")

        return (
            False,
            jnp.empty(
                (self.config.migrate_count, HASH_BYTE_LENGTH[self.config.git_hash]),
                dtype=jnp.uint8,
            ),
            jnp.empty((self.config.migrate_count,)),
        )
