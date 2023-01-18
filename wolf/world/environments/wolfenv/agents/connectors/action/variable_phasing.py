import logging

import numpy as np
from gym.spaces import Discrete

from wolf.world.environments.wolfenv.agents.connectors.action.action_connector import JointActionConnector


class VariablePhasingConnector(JointActionConnector):
    """
    Action Connector which directly selects a green phase of the traffic light **without**
    green phases' order's constraints.
    """

    def __init__(self, connectors_ids, kernel=None):
        self.connectors = [
            _MonoNodeVariablePhasingConnector([node_id], kernel)
            for node_id in connectors_ids
        ]
        self._dim_actions_per_agent = list(map(lambda x: x.num_green_phases, self.connectors))
        action_space = Discrete(np.prod(self._dim_actions_per_agent))
        super().__init__(connectors_ids, action_space, kernel)

        self._LOGGER = logging.getLogger(__name__)

    def a_compute(self, action):
        actions = self._dissolve_action_repr(action, self._dim_actions_per_agent)
        for idx, connector in enumerate(self.connectors):
            connector.compute(actions[idx])

    def _dissolve_action_repr(self, number, bases):
        """
        Dissolve the multi-agent action representation as a list of actions
        corresponding to agents.

        Agents: [center0, center1, center2]
        Dim_of_act: [3, 2, 4]
        self._action_space.n: 3 * 2 * 4 = 24
        Actions -> joint action: [1, 1, 2] -> 8i + 4j + k = 14

        Args:
            number (int or numpy.array): multi-agent action representation
            bases (list[int]): the list of the dimension of action spaces of each agent

        Returns:
            list[int]: the list of actions of each agent

            >>> self._dissolve_action_repr(14, [3, 2, 4])
            >>> [1, 1, 2]
        """
        actions = []
        tmp = number if isinstance(number, int) else number.copy()
        
        for base in bases[::-1]:
            actions.append(tmp % base)
            tmp //= base
        
        return actions[::-1]

    def reset(self):
        for conn in self.connectors:
            conn.reset()


class _MonoNodeVariablePhasingConnector(JointActionConnector):
    def __init__(self, connectors_ids, kernel):
        """
        Creates single node Variable Phasing Connector.

        Args:
            connectors_ids (list): ID of the intersection/node.
            kernel (wolf.world.environments.traffic.kernels.traci_kernel.TraciKernel): Flow kernel.
            
            >>> phases_list: [GREEN, IN_BETWEEN, IN_BETWEEN, GREEN, IN_BETWEEN, GREEN, IN_BETWEEN]
            >>> action_space: Discrete(3)
        """
        self._node_id = connectors_ids[0]
        self.traffic_light = kernel.traffic_lights.get_traffic_light(self._node_id)
        self.num_green_phases = self.traffic_light.len_green_phases()

        action_space = Discrete(self.num_green_phases)
        super().__init__(connectors_ids, action_space, kernel)

        self._LOGGER = logging.getLogger(__name__)
        
    def reset(self):
        self.traffic_light.reset()

    def a_compute(self, action):
        self.traffic_light.switch_to_phase(action)
