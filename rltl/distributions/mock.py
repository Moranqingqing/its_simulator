from rltl.distributions.envs_collection import SampleKwargsEnvsDistribution
import numpy as np

from rltl.envs.mock import MockEnv


class MockDistribution(SampleKwargsEnvsDistribution):

    def __init__(self, backends=None):
        super().__init__(env_prefix="MockEnv",env_cls=MockEnv, backends=backends)

    def _sample_kwargs(self):
        return {"a": np.random.randint(0, 100), "b": np.random.randint(0, 100)}
