from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import weakref

from evox.core import Problem, jit_class, ModuleBase
from evox.operators.mutation import polynomial_mutation
import torch

from phylox import api

logger = logging.getLogger("phylox")

# commit_id is either sha1 - 160 bits or sha256 - 256 bits
# we use 20 bytes to represent the commit id using sha1
# and 32 bytes to represent the commit id using sha256


def array_to_hex(array):
    return array.numpy().tobytes().hex()


def hex_to_array(hex_string):
    return torch.frombuffer(bytearray.fromhex(hex_string), dtype=torch.uint8)


HASH_BYTE_LENGTH = {
    "sha1": 20,
    "sha256": 32,
}


def get_upper_bound(config):
    return torch.full((HASH_BYTE_LENGTH[config.git_hash],), 255, dtype=torch.uint8)


def get_lower_bound(config):
    return torch.zeros((HASH_BYTE_LENGTH[config.git_hash],), dtype=torch.uint8)


def update_branches(config, pop):
    pop = [array_to_hex(individual) for individual in pop]
    api.update_branches(config, pop)
    api.prune_commits(config)


def git_update(config, generation):
    handlers = []
    if generation % config.fetch_every == 0:
        handlers.extend(api.fetch_remote(config))
    if generation % config.push_every == 0:
        handlers.extend(api.push_local_branches(config))

    for proc in handlers:
        proc.wait()


def phylox_git_crossover(config, seeds, pop):
    pop = [array_to_hex(individual) for individual in pop]
    pop_size = len(pop)
    offspring = []
    retry = 0
    for seed in seeds:
        idx1, idx2 = torch.randint(0, pop_size, (2,))
        commit1, commit2 = pop[idx1], pop[idx2]
        while (
            not api.is_novel_merge(config, commit1, commit2)
            and retry < config.max_merge_retry
        ):
            idx1, idx2 = torch.randint(0, pop_size, (2,))
            commit1, commit2 = pop[idx1], pop[idx2]
            retry += 1

        new_commit = api.git_crossover(config, seed.item(), commit1, commit2)
        offspring.append(hex_to_array(new_commit))

    logger.info(f"Git crossover stats: pop_size={pop_size}, retry={retry}")

    return torch.stack(offspring)


def git_crossover(config, pop):
    pop_size, dim = pop.shape
    seeds = torch.randint(0, 1_000_000, (pop_size,))

    return phylox_git_crossover(config, seeds, pop)


def phylox_llm_mutation(config, llm_backend, seeds, pop):
    commits = [array_to_hex(commit) for commit in pop]
    seeds = seeds.tolist()
    new_commits = api.llm_mutation(config, llm_backend, seeds, commits)
    offspring = [hex_to_array(new_commit) for new_commit in new_commits]

    return torch.stack(offspring)


def phylox_llm_crossover(config, llm_backend, seeds, pop):
    commits = [array_to_hex(commit) for commit in pop]
    seeds = seeds.tolist()
    new_commits = api.llm_crossover(config, llm_backend, seeds, commits)
    offspring = [hex_to_array(new_commit) for new_commit in new_commits]

    return torch.stack(offspring)


def llm_mutation(config, llm_backend, pop):
    pop_size = pop.shape[0]
    seeds = torch.randint(0, 1_000_000, (pop_size,))
    return phylox_llm_mutation(config, llm_backend, seeds, pop)


def llm_crossover(config, llm_backend, pop):
    pop_size = pop.shape[0]
    seeds = torch.randint(0, 1_000_000, (pop_size,))
    mating_pool = torch.randint(0, pop_size, (pop_size,))
    pop = pop[mating_pool, :]
    return phylox_llm_crossover(config, llm_backend, seeds, pop)

def load_vectors(config, pop):
    commits = [array_to_hex(commit) for commit in pop]
    vectors = api.load_vectors(config, commits)
    return torch.from_numpy(vectors)

def proxy_vector_mutation(config, pop):
    commits = [array_to_hex(commit) for commit in pop]

    def mut_func(x):
        x = torch.from_numpy(x)
        pop_size, dim = x.shape
        # pro_m = 0.1 * dim
        # x = polynomial_mutation(x, lb=-10, ub=10, pro_m=pro_m)
        x = x + torch.normal(0, 0.1, size=x.shape)
        return x.numpy()

    new_commits = api.vector_mutation(config, commits, mut_func)
    offspring = [hex_to_array(commit) for commit in new_commits]
    return torch.stack(offspring)


__config__ = {}
__llm_backend__ = {}


@jit_class
class GitCrossover(Problem):
    def __init__(self, config):
        super().__init__()
        global __config__
        instance_id = id(self)
        self._index_id_ = instance_id
        if instance_id not in __config__.keys():
            __config__[instance_id] = config
            weakref.finalize(self, __config__.pop, instance_id, None)

    def do(self, parents):
        config = __config__[self._index_id_]
        return git_crossover(config, 1, parents)


@jit_class
class LLMMutation(ModuleBase):
    def __init__(self, config):
        super().__init__()
        global __config__
        global __llm_backend__
        instance_id = id(self)
        self._index_id_ = instance_id
        if instance_id not in __config__.keys():
            __config__[instance_id] = config
            weakref.finalize(self, __config__.pop, instance_id, None)
        if instance_id not in __llm_backend__.keys():
            __llm_backend__[instance_id] = config.llm_backend
            weakref.finalize(self, __llm_backend__.pop, instance_id, None)

    def do(self, pop):
        config = __config__[self._index_id_]
        llm_backend = __llm_backend__[self._index_id_]
        return llm_mutation(config, llm_backend, pop)


@jit_class
class LLMCrossover(ModuleBase):
    def __init__(self, config):
        super().__init__()
        global __config__
        global __llm_backend__
        instance_id = id(self)
        self._index_id_ = instance_id
        if instance_id not in __config__.keys():
            __config__[instance_id] = config
            weakref.finalize(self, __config__.pop, instance_id, None)
        if instance_id not in __llm_backend__.keys():
            __llm_backend__[instance_id] = config.llm_backend
            weakref.finalize(self, __llm_backend__.pop, instance_id, None)

    @torch.compile.disable
    def do(self, pop):
        config = __config__[self._index_id_]
        llm_backend = __llm_backend__[self._index_id_]
        return llm_crossover(config, llm_backend, pop)


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

    illegal_value = 1e8

    commit_to_fitness = {}
    for commit_id, output in zip(unique_pop, outputs):
        performance_cost, time_cost = api.decode_result(output, illegal_value)
        commit_to_fitness[commit_id] = [performance_cost, time_cost]
    fitness = [commit_to_fitness[commit_id] for commit_id in pop]
    fitness = torch.tensor(fitness)
    assert fitness.dtype == torch.float32
    return fitness


__codegen_problem__ = {}


@jit_class
class CodegenProblem(Problem):
    def __init__(self, config):
        super().__init__()
        # ThreadPoolExecutor can achieve parallelism here
        # since the evaluate function will spawn new processes through subprocess.run
        pool = ThreadPoolExecutor(config.evaluate_workers)
        self._index_id_ = id(self)
        global __codegen_problem__
        __codegen_problem__[self._index_id_] = (config, pool)

    @torch.compile.disable
    def evaluate(self, population):
        config, pool = __codegen_problem__[self._index_id_]
        return evaluate(config, pool, population)


@jit_class
class MnistProblem(Problem):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def evaluate(self, population):
        density = torch.sum(population, dim=1) / population.shape[1]
        goodness = torch.ones((population.shape[0], ))
        fitness = torch.stack((density, goodness), dim=1)
        return fitness


def init_population(config, pop_size):
    pop = api.get_initial_branches(config, pop_size)
    pop = [hex_to_array(commit) for commit in pop]
    return torch.stack(pop)


class MigrateHelper:
    def __init__(self, config):
        self.config = config
        self.generation = 0
        self.logger = logging.getLogger("phylox")

    def migrate_from_human(self):
        if self.generation % self.config.human_every == 0:
            commit, fitness = api.migrate_from_human_tags(self.config, 1)

            if commit:
                self.logger.info(f"Found commit by human: {commit}")
                return True, torch.array([hex_to_array(commit[0])]), fitness

        return False, torch.empty(
            (1, HASH_BYTE_LENGTH[self.config.git_hash]), dtype=torch.uint8
        )

    def migrate_from_other_hosts(self):
        if self.generation % self.config.migrate_every == 0:
            commits, fitness = api.migrate_from_other_hosts(
                self.config, self.config.migrate_count
            )

            if len(commits) == self.config.migrate_count:
                self.logger.info(f"Migrating commits from other hosts: {commits}")
                return (
                    True,
                    torch.stack([hex_to_array(commit) for commit in commits]),
                    fitness,
                )

        self.logger.info("No commits found from other hosts")

        return (
            False,
            torch.empty(
                (self.config.migrate_count, HASH_BYTE_LENGTH[self.config.git_hash]),
                dtype=torch.uint8,
            ),
            torch.empty((self.config.migrate_count,)),
        )
