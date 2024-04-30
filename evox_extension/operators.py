import operators
import jax
import jax.numpy as jnp


def git_crossover(key, branch1, branch2):
    seed = jax.random.randint(key, (1, ), 0, jnp.iinfo(jnp.int32).max)