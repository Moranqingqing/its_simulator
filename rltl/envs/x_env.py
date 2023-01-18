import gym
import numpy as np
import math
import gym.spaces as spaces
from gym import Env
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.utils import seeding

import random


class XEnv(Env):

    def __init__(self, x, stochastic=False):
        self.x = x
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.stochastic = stochastic
        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.k = 0
        self.state = -1
        return np.array([self.state])

    def step(self, action):
        if action == 0:
            if self.stochastic and random.random() > 0.5:
                if self.state == -1.:
                    next_state = self.x
                else:
                    next_state = -1.
            else:
                next_state = self.state

        else:
            if self.state == -1.:
                next_state = self.x
            else:
                next_state = -1.

        self.state = next_state
        self.k += 1
        done = self.k > 10

        return np.array([next_state]), next_state, done, {}
