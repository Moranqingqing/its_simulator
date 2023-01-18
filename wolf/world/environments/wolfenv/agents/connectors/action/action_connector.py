from abc import ABC, abstractmethod

from wolf.world.environments.wolfenv.agents.connectors.connector import Connector, AgentListener

"""
Apply action to the simulator
"""
class ActionConnector(Connector,AgentListener):

    def __init__(self, action_space, kernel):
        super().__init__(kernel=kernel)
        self._action_space = action_space
        self._kernel = kernel

    def action_space(self):
        return self._action_space

    @abstractmethod
    def a_compute(self, action):
        raise NotImplementedError

    def compute(self, action):
        if not self._action_space.contains(action):
            raise ValueError("Action {} outside expected value range {}".format(action, self._action_space))
        self.a_compute(action)

"""
Apply a joint action of multiple agents to the simulator
"""
class JointActionConnector(ActionConnector):

    def __init__(self, connectors_ids, action_space, kernel):
        super().__init__(action_space=action_space, kernel=kernel)
        self._connectors_ids = connectors_ids

    @abstractmethod
    def a_compute(self, action):
        raise NotImplementedError


