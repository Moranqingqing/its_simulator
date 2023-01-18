from collections import OrderedDict

from rltl.distributions.envs_collection import EnvsDistribution, FixedSamplingDistribution

import numpy as np

from rltl.envs.mock import MockEnv


class MockFixedDistribution(FixedSamplingDistribution):

    def setup_envs(self):
        envs = OrderedDict({
            "env0": lambda: MockEnv(a=-1, b=1),
            "env1": lambda: MockEnv(a=-2, b=2),
            "env2": lambda: MockEnv(a=-3, b=3)
        })
        return envs
