from abc import ABC, abstractmethod

from gym.spaces import Discrete, Tuple, Box
from ray.rllib.utils.debug import summarize
import numpy as np
from wolf.world.environments.wolfenv.agents.connectors.observation.obs_connector import ObsConnector


class EnvStateConnector(ABC):

    def __init__(self, obs_space, env):
        self._observation_space = obs_space
        self.env = env

    def obs_space(self):
        return self._observation_space

    @abstractmethod
    def _compute(self):
        pass

    def compute(self):
        observation = self._compute()
        if not self._observation_space.contains(observation):
            raise ValueError("Observation\n{}\noutside expected space\n{}"
                             .format(summarize(observation), self._observation_space))

        return observation


#
class AllAgentObservationsEnvState(EnvStateConnector):
    # Did not work earlier due to: https://github.com/ray-project/ray/issues/8407
    # This class is now used by test0_1/qmix.yaml.

    def __init__(self, env):
        obs_space = Tuple([agent.obs_space() for _, agent in env.get_agents().items()])
        EnvStateConnector.__init__(self, env=env, obs_space=obs_space)

    def _compute(self):
        all_obs = []
        for i, agent in self.env._agents.items():
            all_obs.append(agent.observations[-1])

        return tuple(all_obs)

class ConcatSimilarAgentsBoxes(EnvStateConnector):

    def __init__(self, env):
        any_agent = next(iter(env.get_agents().values()))
        local_obs_space = any_agent.obs_space()
        if not isinstance(local_obs_space,Box):
            raise TypeError("Agent local observation should be type Box but is {}".format(type(local_obs_space)))
        obs_space = Box(local_obs_space.low[0],local_obs_space.high[0], (local_obs_space.shape[0] * len(env.get_agents()),))
        EnvStateConnector.__init__(self, env=env, obs_space=obs_space)

    def _compute(self):
        all_obs = []
        for i, agent in self.env._agents.items():
            all_obs.append(agent.observations[-1])

        return np.concatenate(all_obs)



class MockEnvState(EnvStateConnector):
    def __init__(self, env):
        EnvStateConnector.__init__(self, obs_space=Discrete(1), env=None)

    def _compute(self):
        return 0
