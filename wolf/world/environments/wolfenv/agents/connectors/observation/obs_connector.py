from abc import ABC, abstractmethod

from gym.spaces import Box
from ray.rllib.utils.debug import summarize

from wolf.world.environments.wolfenv.agents.connectors.connector import Connector, AgentListener

import numpy as np


class ObsConnector(Connector,AgentListener):

    def __init__(self, observation_space, kernel, **kwargs):
        super().__init__(kernel=kernel)
        # self._observation_class = type(observation_space.sample())
        self._observation_space = observation_space

    def obs_space(self):
        return self._observation_space

    @abstractmethod
    def a_compute(self):
        pass

    def compute(self):
        observation = self.a_compute()

        if not self._observation_space.contains(observation):
            raise ValueError("Observation\n{}\noutside expected space\n{}"
                             .format(summarize(observation), self._observation_space))

        return observation

"""
Joint connector for multiple ObsConnector
"""
class JointObsConnector(ObsConnector):

    def __init__(self, connectors_ids, observation_space, kernel, **kwargs):
        super().__init__(observation_space=observation_space, kernel=kernel,**kwargs)
        self._connectors_ids = connectors_ids

    @abstractmethod
    def a_compute(self):
        pass


class VehConnector(ObsConnector):
    
    def __init__(self, connectors_ids, observation_space, kernel):
        super().__init__(observation_space=observation_space, kernel=kernel)
        self._veh_ids = connectors_ids

    @abstractmethod
    def a_compute(self):
        pass