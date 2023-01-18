import abc
from abc import ABC
import numpy as np
#from tf_agents.trajectories.time_step import TimeStep
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND


class EpsilonSchedule(ABC):
    @abc.abstractmethod
    def compute(self, step):
        raise NotImplementedError

    def plot(self, T):
        import matplotlib.pyplot as plt
        plt.plot(range(T), [self.compute(t) for t in range(T)])
        plt.show()

class ExponentialDecay(EpsilonSchedule):

    def __init__(self, initial_value, decay_rate):
        self.tau = 1. / decay_rate
        self.initial_value = initial_value

    def compute(self, step):
        return self.initial_value * np.exp(-step / self.tau)


class Policy(ABC):

    def __init__(self, action_space):
        self.action_space = action_space

    @abc.abstractmethod
    def act(self, s, batch_mode=False):
        raise NotImplementedError

    def __str__(self):
        return str(self.__class__.__name__)


import tensorflow as tf
#class PolicyFromTfAgent(Policy):

#    def __init__(self, tf_agent_policy):
#        Policy.__init__(self,None)
#        self.tf_agent_policy = tf_agent_policy

#    def act(self, s, batch_mode=False):
#        ss = tf.expand_dims(tf.constant(s,dtype=tf.float32),axis=0)
#        time_step = TimeStep(None, None, None, ss)
#        action_step = self.tf_agent_policy.action(time_step)
#        a = action_step.action.numpy().squeeze()
#        return a


class StaticMaxPolicy(Policy):

    def act(self, s, batch_mode=False):
        if batch_mode:
            return [EXTEND for _ in range(len(s))]
        else:
            return EXTEND


class RandomPolicy(Policy):

    def act(self, s, batch_mode=False):
        if batch_mode:
            return [self.action_space.sample() for _ in range(len(s))]
        else:
            return self.action_space.sample()


class SingleActionPolicy(Policy):

    def __init__(self, a):
        Policy.__init__(self, action_space=None)
        self.a = a

    def act(self, s, batch_mode=False):
        if batch_mode:
            return [self.a for _ in range(len(s))]
        else:
            return self.a


class EpsilonGreedyPolicy(Policy):

    def __init__(self, action_space, greedy_policy):
        Policy.__init__(self, action_space)
        self.greedy_policy = greedy_policy
        self.rdm_policy = RandomPolicy(action_space)

    def set_eps(self, eps):
        self.eps = eps

    def act(self, s, batch_mode=False):
        if np.random.rand() <= self.eps:
            return self.rdm_policy.act(s, batch_mode)
        else:
            return self.greedy_policy.act(s, batch_mode)


if __name__ == "__main__":
    ExponentialDecay(0.5, 0.01).plot(500)
