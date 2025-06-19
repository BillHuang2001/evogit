import torch
from evox.core import Algorithm, Mutable
from evox_extension import (
    init_population,
    evogit_crossover,
    evogit_mutation,
)
import weakref


__config__ = {}
__llm_backend__ = {}


class EvoGitAlgo(Algorithm):
    def __init__(
        self,
        config,
        pop_size: int,
        crossover_every: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.pop_size = pop_size
        self.crossover_every = crossover_every
        if device is None:
            device = torch.get_default_device()
        self.dim = 20
        self.device = device

        population = init_population(config, pop_size)
        self.pop = Mutable(population)

        global __config__
        global __llm_backend__
        instance_id = id(self)
        self._index_id_ = instance_id
        if instance_id not in __config__.keys():
            __config__[instance_id] = config
            weakref.finalize(self, __config__.pop, instance_id, None)
        if instance_id not in __llm_backend__.keys():
            __llm_backend__[instance_id] = config.llm_backend
            weakref.finalize(self, __llm_backend__.pop, instance_id, None)

        self.iter = 0

    def crossover(self, parents):
        global __config__
        global __llm_backend__
        config = __config__[self._index_id_]
        return evogit_crossover(config, parents)

    def mutation(self, pop):
        global __config__
        global __llm_backend__
        config = __config__[self._index_id_]
        llm_backend = __llm_backend__[self._index_id_]
        return evogit_mutation(config, llm_backend, pop)

    def step(self):
        """Perform the optimization step of the workflow."""
        self.iter += 1
        if self.iter % self.crossover_every == 0:
            parents = torch.randperm(self.pop.size(0))
            parents = self.pop[parents]
            offspring = self.crossover(parents.view(self.pop_size // 2, 2, self.dim))
            offspring = torch.repeat_interleave(offspring, repeats=2, dim=0)
            # Evaluate the merge commit by comparing it with both the parents
            comp = self.evaluate(torch.stack([parents, offspring], dim=0))
            # a merge commit is better if it is better than both parents
            comp = comp.view(self.pop_size // 2, 2).all(dim=1).repeat_interleave(2)
            # if the comp is True, it means the offspring is better than the parents
            # so we replace the parents with the offspring
            self.pop = torch.where(comp[:, torch.newaxis], offspring, parents)
        else:
            offspring = self.mutation(self.pop)
            comp = self.evaluate(
                torch.stack([self.pop, offspring], dim=0)
            )  # pairwise comparison
            # Select the best individuals based on the comparison
            self.pop = torch.where(comp[:, torch.newaxis], offspring, self.pop)

    def record_step(self):
        return {"pop": self.pop}
