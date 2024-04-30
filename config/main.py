# the main entry to call the machine written code
# will be call using `python main.py <codedir>`

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
from evox import workflows, monitors, problems

if __name__ == "__main__":
    try:
        # add the codedir to PYTHONPATH
        # and try to import the algorithm
        codedir = sys.argv[1:]
        sys.path.append(codedir)
        import algorithm

        monitor = monitors.EvalMonitor()
        workflow = workflows.StdWorkflow(
            algorithm.Algorithm(), problem=TSP(), monitors=[monitor]
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
        print(json.dumps(result))
    except Exception as e:
        import traceback
        result = {
            "status": "error",
            "stack_trace": traceback.format_stack(),
            "best_fit": None,
            "time_cost": None,
        }
        print(json.dumps(result))