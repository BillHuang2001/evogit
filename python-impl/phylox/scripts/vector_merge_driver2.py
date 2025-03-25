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
child_vec = ancestor_vec + delta1 + delta2

# Resolve conflicts
## Find the conflicting items
conflict = (delta1 != 0) & (delta2 != 0) # conflict if both parents try to change the same item
conflict_vec1 = delta1[conflict]
conflict_vec2 = delta2[conflict]
## Random linear combination
rng = np.random.default_rng()
factor = rng.uniform(0, 1, conflict_vec1.shape)
conflict_resolved = conflict_vec1 * factor + conflict_vec2 * (1 - factor)
## Overwrite the conflicting items
child_vec[conflict] = conflict_resolved

np.save(parent1, child_vec)
