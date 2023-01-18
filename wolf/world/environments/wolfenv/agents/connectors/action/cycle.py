from copy import deepcopy

from numpy import binary_repr

from wolf.world.environments.wolfenv.agents.connectors.action.action_connector import JointActionConnector
from wolf.world.environments.wolfenv.phase import Phase
import gym
import logging
from itertools import cycle
import numpy as np
import math


def mapping(cardinality, n):
    from itertools import permutations
    phases = list(range(cardinality))
    perm = permutations(phases)
    res = next(perm)
    for _ in range(n):
        res = next(perm)
    
    return list(res)


class CycleConnector(JointActionConnector):
    # ATTENTION: this controller is not ready to use
    def __init__(self, connectors_ids, kernel=None):
        action_space_dict = {}
        self.connectors = []
        for node_id in connectors_ids:
            self.connectors.append(_MonoNodeCycleConnector(
                connectors_ids=[node_id],
                kernel=kernel))
            action_space_dict[node_id] = self.connectors[-1].action_space

        if len(connectors_ids) == 1:
            super().__init__(action_space=self.connectors[0].action_space, kernel=kernel, connectors_ids=connectors_ids)
        else:        
            super().__init__(action_space=gym.spaces.Dict(action_space_dict), kernel=kernel, connectors_ids=connectors_ids)
        self._LOGGER = logging.getLogger(__name__)

    def a_compute(self, action):
        for connector, node_id in zip(self.connectors, self._connectors_ids):
            connector.compute(action[node_id])

    def reset(self):
        for conn in self.connectors:
            conn.reset()


class _MonoNodeCycleConnector(JointActionConnector):
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
        
        order = gym.spaces.Box(0, math.factorial(self.__n_green) - 1, shape=(1, ), dtype=np.uint8)
        length = gym.spaces.Box(0, 1, shape=(self.__n_green, ))
        self.action_space = gym.spaces.Dict({"order": order, "length": length})

        super().__init__(connectors_ids=connectors_ids, action_space=self.action_space, kernel=kernel)
        self._LOGGER = logging.getLogger(__name__)

    def reset(self):
        self.traffic_light.reset()

    def a_compute(self, action):

        action["order"] = mapping(cardinality=self.__n_green,
                                  n=action["order"])
                                                
        self.traffic_light.set_cycle(action)


