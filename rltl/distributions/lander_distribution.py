from rltl.distributions.envs_collection import SampleKwargsEnvsDistribution
import numpy as np

from rltl.envs.lander import ParametricLunarLanderEnv


class LunarLanderDistribution(SampleKwargsEnvsDistribution):

    def __init__(self, backends=None):
        super().__init__(env_prefix="LunarLander", env_cls=ParametricLunarLanderEnv, backends=backends)

    def _sample_kwargs(self):
        # will need to think about this - in [0, 2] is not large enough of a density range I think
        return {"density": (np.random.rand() * 2) * ParametricLunarLanderEnv.DEFAULT_DENSITY}
