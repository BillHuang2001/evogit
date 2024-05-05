import jax.numpy as jnp
from evox import Algorithm, State
from evox.operators import crossover, mutation, selection
from jax import random


class TSPBaselineGA(Algorithm):
    def __init__(self, pop_size, num_cites, num_offspring):
        super().__init__()
        self.pop_size = pop_size
        self.num_cities = num_cites
        self.num_offsprings = num_offspring

    def setup(self, key):
        key, subkey = random.split(key)
        population = jnp.tile(jnp.arange(self.num_cities), (self.pop_size, 1))
        population = random.permutation(subkey, population, axis=1, independent=True)
        fitness = jnp.full((self.pop_size,), jnp.inf)
        offspring = jnp.empty((self.num_offspring, self.num_cities), dtype=int)
        return State(
            population=population, fitness=fitness, offspring=offspring, key=key
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        return state.update(fitness=fitness)

    def ask(self, state):
        key, sel_key, mut_key = random.split(state.key, 3)
        selected = random.choice(sel_key, state.population.shape[0])
        offspring = random.permutation(
            mut_key, state.population[selected], axis=1, independent=True
        )
        state = state.update(key=key, offspring=offspring)
        return offspring, state

    def tell(self, state, fitness):
        population = jnp.concatenate([state.population, state.offsprings], axis=0)
        fitness = jnp.concatenate([state.fitness, fitness], axis=0)
        population, fitness = selection.topk_fit(population, fitness, self.pop_size)
        return state.update(population=population, fitness=fitness)
