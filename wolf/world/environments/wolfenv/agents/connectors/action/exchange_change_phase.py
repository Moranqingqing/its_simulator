from copy import deepcopy
import logging
from itertools import cycle

from wolf.world.environments.wolfenv.agents.connectors.action.action_connector import JointActionConnector
from wolf.world.environments.wolfenv.phase import Phase
from wolf.world.environments.wolfenv.kernels.tl_wolf_kernel import AvailActionsMethod
import numpy as np
import gym
from gym.spaces import Discrete


EXTEND = 0
CHANGE = 1
NOOP = 2


class ExtendChangePhaseConnector(JointActionConnector):

    # we should be able to define different phase, and tl_params for each intersection, but here, it is copy pasted
    def __init__(self, connectors_ids, add_noop_action=False, kernel=None):
        num_nodes = len(connectors_ids)
        if add_noop_action:
            action_space = Discrete(3 ** num_nodes)
            self.avail_actions_method = AvailActionsMethod.EXTEND_CHANGE
        else:
            action_space = Discrete(2 ** num_nodes)
            self.avail_actions_method = None

        super().__init__(connectors_ids, action_space, kernel)
        self._LOGGER = logging.getLogger(__name__)

        self.connectors = [
            _MonoNodeExtendChangePhaseConnector([node_id], add_noop_action, kernel)
            for node_id in connectors_ids
        ]
        self.num_actions_per_agent = int(self._action_space.n / len(self.connectors))

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


class _MonoNodeExtendChangePhaseConnector(JointActionConnector):
    def __init__(self, connectors_ids, add_noop_action, kernel):
        """
        Creates single node Extend Change Connector.

        Args:
            connectors_ids (list): ID of the intersection/node.
            add_noop_action (bool): Whether NOOP action is added or not.
            kernel (wolf.world.environments.wolfenv.kernels.traci_wolf_kernel.TraciWolfKernel): Flow kernel.
        """
        action_space = Discrete(3) if add_noop_action else Discrete(2)
        super().__init__(connectors_ids, action_space, kernel)

        self._LOGGER = logging.getLogger(__name__)
        self._node_id = connectors_ids[0]
        self.traffic_light = self._kernel.traffic_lights.get_traffic_light(self._node_id)

    def reset(self):
        self.traffic_light.reset()

    def a_compute(self, action):
        if action == EXTEND:
            self.traffic_light.extend()
        elif action == CHANGE:
            self.traffic_light.change()
        elif action == NOOP:
            self.traffic_light.noop()
        else:
            raise ValueError(f'Unexpected action={action} recieved.')
