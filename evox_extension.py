import warnings

import jax
from jax.experimental import io_callback
import jax.numpy as jnp

import jax
import jax.numpy as jnp
from gaea import api
from functools import partial
from evox import Monitor, Problem, jit_class


def array_to_hex(array):
    return array.tobytes().hex()


def hex_to_array(hex_string):
    return jnp.frombuffer(bytes.fromhex(hex_string), dtype=jnp.uint8)


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
    tag1, tag2 = parents
    offspring = []
    for seed, tag1, tag2 in zip(seeds, parents[0], parents[1]):
        tag1 = array_to_hex(tag1)
        tag2 = array_to_hex(tag2)
        new_tag = api.git_crossover(config, seed, tag1, tag2)
        print(new_tag)
        offspring.append(hex_to_array(new_tag))

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

    return_type = jax.ShapeDtypeStruct((num_pairs, 16), jnp.uint8)
    return io_callback(partial(gaea_git_crossover, config), return_type, seeds, parents)


def gaea_llm_mutation(config, llm_backend, seeds, pop):
    tags = [array_to_hex(tag) for tag in pop]
    new_tags = api.llm_mutation(config, llm_backend, seeds, tags)
    offspring = [hex_to_array(new_tag) for new_tag in new_tags]

    return jnp.stack(offspring)


@partial(jax.jit, static_argnums=(0, 1))
def llm_mutation(config, llm_backend, key, pop):
    pop_size = pop.shape[0]
    seeds = jax.random.randint(key, (pop_size,), 0, jnp.iinfo(jnp.int32).max)
    return_type = jax.ShapeDtypeStruct((pop_size, 16), jnp.uint8)
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
    fitness = [api.evaluate(config, tag, []) for tag in pop]
    return -jnp.array(fitness)


@jit_class
class CodegenProblem(Problem):
    def __init__(self, config):
        self.config = config

    def evaluate(self, state, population):
        return_dtype = jax.ShapeDtypeStruct((population.shape[0],), jnp.float32)
        return (
            io_callback(partial(evaluate, self.config), return_dtype, population),
            state,
        )


def init_population(config, pop_size):
    pop = api.get_initial_branches(config, pop_size)
    pop = [hex_to_array(tag) for tag in pop]
    return jnp.stack(pop)
