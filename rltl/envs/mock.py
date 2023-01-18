import gym
import numpy as np

class MockEnv(gym.Env):
    def __init__(self, **kwargs):
        self._i = None
        self.a = int(kwargs["a"]) # better than a or b, so we can pass dictionary
        self.b = int(kwargs["b"])
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(1,),
            dtype=np.float32)
    
    def seed(self, seed=None):
        return [seed]

    def reset(self):
        self._i = 0
        obs = np.zeros((1,), dtype=np.float32)
        return obs

    def step(self, action):
        obs = np.zeros((1,), dtype=np.float32)
        reward = 0
        done = self._i >= 9
        infos = {}
        self._i += 1
        return obs, reward, done, infos
