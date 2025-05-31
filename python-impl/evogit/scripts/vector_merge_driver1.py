#!/usr/bin/env python
"""Usage: python vector_merge_driver.py <ancestor> <parent1> <parent2>"""
import sys
import numpy as np

# Read file paths
ancestor = sys.argv[1]
parent1 = sys.argv[2]
parent2 = sys.argv[3]

# Load vectors
ancestor_vec = np.load(ancestor)
parent1_vec = np.load(parent1)
parent2_vec = np.load(parent2)
assert ancestor_vec.shape == parent1_vec.shape == parent2_vec.shape

# Calculate delta vectors
delta1 = parent1_vec - ancestor_vec
delta2 = parent2_vec - ancestor_vec

# Merge vectors
## If there is no conflict, one of the delta is zero, so the other delta is applied
## If there is a conflict, it will be handled by a random linear combination in the next step
rng = np.random.default_rng()
factor1 = rng.uniform(0, 1, delta1.shape)
factor2 = rng.uniform(0, 1, delta2.shape)
child_vec = ancestor_vec + factor1 * delta1 + factor2 * delta2

np.save(parent1, child_vec)
