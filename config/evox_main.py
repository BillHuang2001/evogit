# the main entry to call the machine written code
# will be call using `python main.py <codedir>`

import json
import sys
import time
import traceback

import jax
import jax.numpy as jnp
from jax import vmap
from evox import workflows, monitors, Problem, jit_class


def l2_distance(city1, city2):
    return jnp.sum((city1 - city2) ** 2) ** 0.5


@jit_class
class TSP(Problem):
    """Travelling salesman problem"""

    def __init__(self, cities):
        """
        There are two ways to contruct a TSP problem.
        1. Using a list of cities' coordinates.
        2. Using a dist_matrix
        """
        self.cities = cities  # (N, M) where N is the number of cities, M is dimension of the coordinates.

    def _sum_distance(self, x):
        start = x
        end = jnp.roll(x, shift=1, axis=0)
        dist = vmap(l2_distance)(self.cities[start], self.cities[end])
        return jnp.sum(dist)

    def evaluate(self, state, X):
        return vmap(self._sum_distance)(X), state


if __name__ == "__main__":
    try:
        # add the cwd to PYTHONPATH
        # and try to import the algorithm
        sys.path.append("./")
        from algorithm import TSPGA

        pop_size = 100
        num_cities = 50
        num_offspring = 100

        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        cities = jax.random.randint(key, (num_cities, 2), minval=0, maxval=100)
        problem = TSP(cities)
        monitor = monitors.EvalMonitor()

        workflow = workflows.StdWorkflow(
            TSPGA(
                pop_size=pop_size, num_cities=num_cities, num_offspring=num_offspring
            ),
            problem=problem,
            monitors=[monitor],
            opt_direction="min",
        )

        state = workflow.init(key)

        state = workflow.step(state)

        start = time.perf_counter()
        for i in range(100):
            state = workflow.step(state)

        monitor.close()
        best_fitness = monitor.get_best_fitness().item()
        end = time.perf_counter()
        result = {
            "status": "finished",
            "stack_trace": "",
            "best_fit": best_fitness,
            "time_cost": end - start,
        }
    except Exception as e:
        result = {
            "status": "error",
            "stack_trace": traceback.format_exc(),
            "best_fit": None,
            "time_cost": None,
        }
    finally:
        print(json.dumps(result))
