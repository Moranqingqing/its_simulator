from abc import ABC, abstractmethod
import numpy as np
from gym.spaces import Discrete
from gym.spaces.box import Box

from wolf.world.environments.wolfenv.agents.connectors.connector import Connector, AgentListener


class RewardConnector(Connector,AgentListener):

    def __init__(self, reward_space, kernel):
        super().__init__(kernel=kernel)
        self._reward_class = type(reward_space.sample())
        self._reward_space = reward_space

    def reward_space(self):
        return self._reward_space

    def compute(self):
        reward = self.a_compute()
        if reward is None:
            raise ValueError("Reward is None")
        # if self._reward_class is not None and not isinstance(reward, self._reward_class):
        #     raise TypeError("reward should be instance of {} but is {} instead"
        #                     .format(self._reward_class, type(reward)))

        if not self._reward_space.contains(reward):
            raise ValueError("Reward {} outside expected value range {}"
                             .format(reward, self._reward_space))
        return reward

    @abstractmethod
    def a_compute(self):
        raise NotImplementedError



"""
Proceed the result of multiples rewardConnector
"""
class JointRewardConnector(RewardConnector):

    def __init__(self, connectors_ids, reward_space, kernel):
        super().__init__(reward_space=reward_space, kernel=kernel)
        self._connectors_ids = connectors_ids

    @abstractmethod
    def a_compute(self):
        raise NotImplementedError


class AccumulateSumRewardConnector(JointRewardConnector):
    def __init__(self, connectors_ids, kernel, connectors):
        self.connectors = connectors
        if isinstance(connectors[0].reward_space(), Discrete):
            n_agg = 0
            for con in connectors:
                n_agg += con.reward_space().n
            reward_space = Discrete(n_agg)
        elif isinstance(connectors[0].reward_space(), Box):
            low = -np.inf
            high = np.inf
            # for con in connectors:
            #     low += con.reward_space().low
            #     high += con.reward_space().high
            reward_space = Box(low, high, ())
        else:
            raise Exception("Can't add this kind of reward spaces: {}".format(type(connectors[0].reward_space())))

        super().__init__(connectors_ids=connectors_ids, reward_space=reward_space, kernel=kernel)

    def a_compute(self):
        reward = 0
        for conn in self.connectors:
            reward += conn.compute()
        return reward

    def reset(self):
        for conn in self.connectors:
            conn.reset()
