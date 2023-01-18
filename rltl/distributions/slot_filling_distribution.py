from rltl.distributions.envs_collection import SampleKwargsEnvsDistribution
import numpy as np

from rltl.envs.cartpole import ParametricCartPoleEnv


class SlotFillingDistribution(SampleKwargsEnvsDistribution):

    def __init__(self, backends=None):
        super().__init__(env_prefix="SlotFilling", env_cls=ParametricCartPoleEnv, backends=backends)

    def _sample_kwargs(self):
        return {"user_params": {"cerr": -1, "cok": 1, "ser": 0.5, "cstd": 0.2, "proba_hangup": np.random.rand()}}
