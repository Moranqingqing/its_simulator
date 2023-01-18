from abc import ABC, abstractmethod

from wolf.world.environments.wolfenv.agents.connectors.connector import Connector, AgentListener


class DoneConnector(Connector,AgentListener):

    def __init__(self,kernel=None):
        super().__init__(kernel=kernel)
        self._kernel= kernel

    @abstractmethod
    def compute(self):
        """ must return a boolean"""
        raise NotImplementedError
