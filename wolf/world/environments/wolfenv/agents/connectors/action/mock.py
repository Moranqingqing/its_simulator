from wolf.world.environments.wolfenv.agents.connectors.action.action_connector import ActionConnector
import gym


class MockActionConnector(ActionConnector):

    def __init__(self, **kwargs):
        super().__init__(action_space=gym.spaces.Discrete(1), kernel=None)

    def a_compute(self, action):
        pass
