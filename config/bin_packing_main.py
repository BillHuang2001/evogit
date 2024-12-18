import json
import sys
import traceback
import random
import os
import copy


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


NUM_TEST_CASES = 10


def valid_plan(data, plan):
    if plan is None or not isinstance(plan, list):
        raise ValueError(f"Plan must be a list, got {type(plan)}")

    if len(plan) != len(data):
        raise ValueError(f"Plan length must be equal to data length")

    if min(plan) != 0:
        raise ValueError(f"Bin index must start from 0")

    bins = [0 for _ in range(max(plan) + 1)]
    for item, bin_id in zip(data, plan):
        bins[bin_id] += item
        if bins[bin_id] > 1:
            raise ValueError(f"Bin {bin_id} has sum > 1")
    return True


if __name__ == "__main__":
    try:
        # add the cwd to PYTHONPATH
        # and try to import the algorithm
        sys.path.append("./")
        with HiddenPrints():
            from algorithm import bin_packing

        # set the random seed
        random.seed(42)
        fitness = []
        with HiddenPrints():
            for _ in range(NUM_TEST_CASES):
                num_bins = 100_000
                input_data = [random.random() for _ in range(num_bins)]
                plan = bin_packing(copy.copy(input_data)) # copy data to avoid mutation
                valid_plan(input_data, plan)
                fitness.append(max(plan) + 1)

        result = {
            "status": "finished",
            "stack_trace": "",
            "fitness": sum(fitness) / len(fitness),
        }
    except Exception as e:
        filtered_tb = []
        for frame in traceback.extract_tb(e.__traceback__):
            if (
                "algorithm.py" in frame.filename
                or "bin_packing_main.py" in frame.filename
            ):
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
