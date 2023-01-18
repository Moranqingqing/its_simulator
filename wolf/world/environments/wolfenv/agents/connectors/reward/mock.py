from gym.spaces import Discrete

from wolf.world.environments.wolfenv.agents.connectors.reward.reward_connector import RewardConnector


class MockRewardConnector(RewardConnector):

    def __init__(self, **kwargs):
        super().__init__(reward_space=Discrete(100), kernel=None)

    def a_compute(self):
        # return self._reward_space.sample()
        return 1



