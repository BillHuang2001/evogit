# the main entry to call the machine written code
# will be call using `python main.py <codedir>`

import json
import sys
import time
from uuid import uuid1
import traceback

import jax
import jax.numpy as jnp
from jax import vmap
from evox import workflows, monitors, Problem, jit_class, Monitor


def l2_distance(city1, city2):
    return jnp.sum((city1 - city2) ** 2) ** 0.5


@jax.jit
def is_correct_permutation(x):
    # x is a 1 dim vector
    visited = jnp.zeros(x.shape[0], dtype=jnp.bool_)
    visited = visited.at[x].set(True)
    return jnp.all(visited) & jnp.all(visited >= 0) & jnp.all(visited < x.shape[0])


@jax.jit
def population_is_correct(population):
    return vmap(is_correct_permutation)(population).all()


def call_concorde(cities, distance):
    import os

    n = cities.shape[0]
    tsp_file = f"""NAME : tsp_example
TYPE : TSP
DIMENSION : {n}
EDGE_WEIGHT_TYPE : {"EUC_2D" if distance == "l2" else "GEOM"}
NODE_COORD_SECTION"""
    cities = [f"{i+1} {x} {y}" for i, (x, y) in enumerate(cities)]
    tsp_file += "\n" + "\n".join(cities)
    uuid = uuid1().hex
    with open(f"./temp/{uuid}.tsp", "w") as f:
        f.write(tsp_file)

    os.system(f"/home/bchuang/llm/tsp_solver/solver ./temp/{uuid}.tsp")
    with open(f"{uuid}.sol", "r") as f:
        sol = f.read()
    os.remove(f"./temp/{uuid}.tsp")
    os.remove(f"{uuid}.sol")
    os.remove(f"{uuid[:20]}")
    os.remove(f"O{uuid[:20]}")
    sol = sol.split("\n")
    index = sol.index(str(n))
    sol = sol[index + 1 :]
    sol = [line.split() for line in sol]
    sol = [int(item) for sublist in sol for item in sublist]
    return sol


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

    def _calc_exact_solution(self):
        sol = call_concorde(self.cities, self.distance)
        return jnp.array(sol)


class CheckTSPMonitor(Monitor):
    def __init__(self, num_cities, num_offspring):
        super().__init__()
        self.num_cities = num_cities
        self.num_offspring = num_offspring

    def hooks(self):
        return ["post_ask"]

    def post_ask(self, state, cand_sol):
        assert cand_sol.shape == (self.num_offspring, self.num_cities)
        all_valid = population_is_correct(cand_sol)
        jax.experimental.io_callback(self._assert, None, all_valid)

    def _assert(self, all_valid):
        assert all_valid, "The output is not a valid permutation. A permutation should contain all integers from 0 to N-1 exactly once."


if __name__ == "__main__":
    try:
        # add the cwd to PYTHONPATH
        # and try to import the algorithm
        sys.path.append("./")
        from algorithm import TSPGA

        pop_size = 128
        num_cities = 100
        num_offspring = 128

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
            monitors=[CheckTSPMonitor(num_cities, num_offspring), monitor],
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
            "fitness": best_fitness,
            "time_cost": end - start,
        }
    except Exception as e:
        filtered_tb = []
        for frame in traceback.extract_tb(e.__traceback__):
            if "algorithm.py" in frame.filename or "evox_main.py" in frame.filename:
                filtered_tb.append(frame)
        exception_msg = traceback.format_exception_only(type(e), e)
        filtered_tb = traceback.format_list(filtered_tb) + exception_msg
        result = {
            "status": "error",
            "stack_trace": "".join(filtered_tb),
            "fitness": None,
            "time_cost": None,
        }
    finally:
        print(json.dumps(result))
