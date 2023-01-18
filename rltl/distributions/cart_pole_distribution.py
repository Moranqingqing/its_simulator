from rltl.distributions.envs_collection import SampleKwargsEnvsDistribution
import numpy as np

from rltl.envs.cartpole import ParametricCartPoleEnv


class CartPoleDistribution(SampleKwargsEnvsDistribution):

    def __init__(self, backends=None):
        super().__init__(env_prefix="CartPole", env_cls=ParametricCartPoleEnv, backends=backends)

    def _sample_kwargs(self):
        return {"gravity": np.random.rand() * 2 * ParametricCartPoleEnv.DEFAULT_GRAVITY}

class PoleLengthCartPoleDistribution(SampleKwargsEnvsDistribution):

    def __init__(self, backends=None):
        super().__init__(env_prefix="CartPole", env_cls=ParametricCartPoleEnv, backends=backends)

    def _sample_kwargs(self):
        return {"length": np.random.rand() * 2 * ParametricCartPoleEnv.DEFAULT_LENGTH}