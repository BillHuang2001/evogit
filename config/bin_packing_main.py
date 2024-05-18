import json
import sys
import time
import traceback
import random


NUM_TEST_CASES = 10
NUM_MAX_BINS = 1000

def valid_plan(data, plan):
    if plan is None or not isinstance(plan, list):
        return False

    bins = [[] for _ in range(max(plan) + 1)]
    for item, bin_id in zip(data, plan):
        bins[bin_id].append(item)
    for bin_ in bins:
        if sum(bin_) > 1:
            return False
    return True


if __name__ == "__main__":
    try:
        # add the cwd to PYTHONPATH
        # and try to import the algorithm
        sys.path.append("./")
        from algorithm import bin_packing

        random.seed(42)
        fitness = []
        for _ in range(NUM_TEST_CASES):
            num_bins = random.randint(2, NUM_MAX_BINS)
            data = [random.random() for _ in range(num_bins)]
            plan = bin_packing(data)
            assert plan is not None, "The return value of the algorithm is None."
            if not valid_plan(data, plan):
                fitness.append(1e6)
            else:
                fitness.append(max(plan) + 1)

        result = {
            "status": "finished",
            "stack_trace": "",
            "fitness": sum(fitness),
        }
    except Exception as e:
        result = {
            "status": "error",
            "stack_trace": traceback.format_exc(),
            "fitness": fitness,
        }
    finally:
        print(json.dumps(result))