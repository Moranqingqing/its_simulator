from copy import deepcopy

from numpy import binary_repr

from wolf.world.environments.wolfenv.agents.connectors.action.action_connector import JointActionConnector
from wolf.world.environments.wolfenv.phase import Phase
import gym
import logging
from itertools import cycle
import numpy as np


class FixedOrderCycleConnector(JointActionConnector):
    # TODO: write the mapping function
    def __init__(self, connectors_ids, kernel=None):
        self.connectors = []
        self.indexes = []
        total_len = 0
        for node_id in connectors_ids:
            self.connectors.append(_MonoNodeFixedOrderCycleConnector(
                connectors_ids=[node_id],
                kernel=kernel))
            self.indexes.append([total_len, self.connectors[-1].get_shape()])
            total_len += self.connectors[-1].get_shape()
        
        super().__init__(action_space=gym.spaces.Box(0, 1, (total_len, )), kernel=kernel, connectors_ids=connectors_ids)
        self._LOGGER = logging.getLogger(__name__)

    def a_compute(self, action):
        for connector, idx in zip(self.connectors, self.indexes):
            connector.compute(action[idx[0]:idx[1]])

    def reset(self):
        for conn in self.connectors:
            conn.reset()


class _MonoNodeFixedOrderCycleConnector(JointActionConnector):
    def __init__(self, connectors_ids, kernel):
        """

        Parameters
        ----------
        connectors_ids ids of the nodes where the traffic lights are
        kernel flow kernel
        """
        self._node_id = connectors_ids[0]
        self.traffic_light = kernel.traffic_lights.get_traffic_light(
            node_id=self._node_id)
        self.__n_green = self.traffic_light.num_green_phases()

        self.action_space = gym.spaces.Box(0, 1, shape=(self.__n_green, ))
        super().__init__(connectors_ids=connectors_ids,
                         action_space=self.action_space,
                         kernel=kernel)
        self._LOGGER = logging.getLogger(__name__)

    def get_shape(self):
        return self.__n_green

    def reset(self):
        self.traffic_light.reset()

    def a_compute(self, action):                                                
        self.traffic_light.set_cycle(action)
