# the main entry to call the machine written code
# will be call using `python main.py <codedir>`

import json
import sys
import time
import traceback

import jax
import jax.numpy as jnp
from jax import vmap
from evox import workflows, monitors, problems, Problem, jit_method


def l2_distance(city1, city2):
    return jnp.sum((city1 - city2) ** 2) ** 0.5


class TSP(Problem):
    """Travelling salesman problem"""

    def __init__(self, cities, dist_matrix, distance="l2"):
        """
        There are two ways to contruct a TSP problem.
        1. Using a list of cities' coordinates.
        2. Using a dist_matrix
        """
        self.cities = cities  # (N, M) where N is the number of cities, M is dimension of the coordinates.
        self.dist_matrix = dist_matrix
        self.distance = distance
        self.calc_distance = l2_distance

    def _sum_distance(self, x):
        start = x
        end = jnp.roll(x, shift=1, axis=0)
        if self.cities is not None:
            dist = vmap(self.calc_distance)(self.cities[start], self.cities[end])
            return jnp.sum(dist)
        else:
            return jnp.sum(self.dist_matrix[start][end])

    @jit_method
    def evaluate(self, state, X):
        return vmap(self._sum_distance)(X), state


if __name__ == "__main__":
    try:
        # add the cwd to PYTHONPATH
        # and try to import the algorithm
        sys.path.append("./")
        import algorithm

        monitor = monitors.EvalMonitor()
        problem = TSP()
        workflow = workflows.StdWorkflow(
            algorithm.Algorithm(),
            problem=TSP(),
            monitors=[monitor],
            opt_direction="min",
        )

        key = jax.random.PRNGKey(0)
        state = workflow.init(key)

        best_fitness = monitor.get_best_fitness()
        result = {
            "status": "finished",
            "stack_trace": "",
            "best_fit": best_fitness,
            "time_cost": time,
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
