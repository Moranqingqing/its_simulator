from copy import deepcopy
import logging
from itertools import cycle

from wolf.world.environments.wolfenv.agents.connectors.action.action_connector import JointActionConnector
from wolf.world.environments.wolfenv.phase import Phase
from wolf.world.environments.wolfenv.kernels.tl_wolf_kernel import AvailActionsMethod
import numpy as np
import gym
from gym.spaces import Discrete


class PhaseSelectConnector(JointActionConnector):
    """
    Action Connector which directly selects the phase of the wolfenv light.
    With it avail_actions are also provided. avail_actions can be configured to limit the phase order or the phase constraints.
    By default both are applied.
    """

    def __init__(self, connectors_ids, kernel=None):
        self.avail_actions_method = AvailActionsMethod.PHASE_SELECT

        self.connectors = [
            _MonoNodePhaseSelectConnector([node_id], kernel)
            for node_id in connectors_ids
        ]
        action_space = Discrete(sum(map(lambda x: x.num_green_phases, self.connectors)))
        super().__init__(connectors_ids, action_space, kernel)
        self.num_actions_per_agent = int(self._action_space.n / len(self.connectors))
        self._LOGGER = logging.getLogger(__name__)

    def a_compute(self, action):
        rep = self.concat_action_repr(action, self.num_actions_per_agent, len(self.connectors))
        for i, connector in enumerate(self.connectors):
            action_int = int(rep[i])
            connector.compute(action_int)

    def concat_action_repr(self, number, base, width):
        return np.base_repr(number, base, width)[-width:]

    def reset(self):
        for conn in self.connectors:
            conn.reset()

    def get_avail_actions(self):
        avail_actions = np.concatenate([c.get_avail_actions() for c in self.connectors])
        return avail_actions


class _MonoNodePhaseSelectConnector(JointActionConnector):
    def __init__(self, connectors_ids, kernel):
        """
        Creates single node Phase Select Connector.
        avail_actions enforce both phase ordering and minimum and maximum duration of the phase.

        Args:
            connectors_ids (list): ID of the intersection/node.
            kernel (wolf.world.environments.wolfenv.kernels.traci_wolf_kernel.TraciWolfKernel): Flow kernel.
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
        if self._action_space.contains(action):
            self.traffic_light.select_phase_by_index(action)
        else:
            raise ValueError(f'Unexpected action={action} recieved.')
