from typing import Any
import warnings

import jax
import jax.numpy as jnp

from evox import Monitor


class PopMonitor(Monitor):
    def __init__(
        self,
        population_name="population",
        fitness_name="fitness",
        to_host=False,
        fitness_only=False,
    ):
        super().__init__()
        self.population_name = population_name
        self.fitness_name = fitness_name
        self.to_host = to_host
        self.population_history = []
        self.fitness_history = []
        self.fitness_only = fitness_only

    def hooks(self):
        return ["post_step"]

    def post_step(self, state):
        if not self.fitness_only:
            population = getattr(
                state.get_child_state("algorithm"), self.population_name
            )
            if self.to_host:
                population = jax.device_put(population, self.host)
            self.population_history.append(population)

        fitness = getattr(state.get_child_state("algorithm"), self.fitness_name)
        if self.to_host:
            fitness = jax.device_put(fitness, self.host)
        self.fitness_history.append(fitness)

    def get_population_history(self):
        return self.population_history

    def get_fitness_history(self):
        return self.fitness_history

