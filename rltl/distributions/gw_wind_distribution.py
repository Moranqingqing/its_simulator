from rltl.distributions.envs_collection import SampleKwargsEnvsDistribution
import numpy as np

from rltl.envs.cartpole import ParametricCartPoleEnv
from rltl.envs.gridworld.envgridworld import EnvGridWorld


class GridWorldDistribution(SampleKwargsEnvsDistribution):

    def __init__(self,env_prefix,env_cls_or_factory, backends=None):
        super().__init__(env_prefix=env_prefix, env_cls_or_factory=env_cls_or_factory, backends=backends)

    def _sample_kwargs(self):
        return {
            "wind": (-1 + 2 * np.random.rand(), -1 + 2 * np.random.rand())
        }
