from gym.spaces import Tuple, Dict, Discrete, Box
import numpy as np

import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.qmix.qmix_policy import ENV_STATE

action_space = Discrete(2)
agent_obs_space = Box(0, 1, (3,))

observation_space = Dict({
    "obs": agent_obs_space,
    # ENV_STATE: Box(0, 1, (6,))
    ENV_STATE: Tuple((agent_obs_space, agent_obs_space))
})
grouped_obs_space = Tuple([observation_space, observation_space])
grouped_action_space = Tuple([action_space, action_space])


class TwoStepGame(MultiAgentEnv):

    def __init__(self, config):
        self.state = None
        self.agent_1 = 0
        self.agent_2 = 1
        self.observation_space = grouped_obs_space
        self.k = 0

    def reset(self):
        self.k = 0
        return self.obs()

    def step(self, action_dict):
        rewards = {
            self.agent_1: 1.0,
            self.agent_2: 1.0
        }
        self.k += 1
        done = self.k >= 10
        dones = {"__all__": done}
        infos = {}
        return self.obs(), rewards, dones, infos

    def obs(self):
        obs_agent_1 = [0.1, 0.2, 0.3]
        obs_agent_2 = [0.4, 0.5, 0.6]
        state = tuple([obs_agent_1, obs_agent_2])
        # state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        obs = {
            self.agent_1: {
                "obs": obs_agent_1,
                ENV_STATE: state
            },
            self.agent_2: {
                "obs": obs_agent_2,
                ENV_STATE: state
            }
        }
        return obs


if __name__ == "__main__":
    grouping = {
        "group_1": [0, 1],
    }

    register_env(
        "grouped_twostep",
        lambda config: TwoStepGame(config).with_agent_groups(
            grouping, obs_space=grouped_obs_space, act_space=grouped_action_space))

    config = {
        "num_workers": 0,
        "mixer": "qmix",
        "env_config": {}
    }

    ray.init(num_cpus=2 or None, local_mode=True)
    tune.run(
        "QMIX",
        stop={
            "timesteps_total": 10,
        },
        config=dict(config, **{
            "env": "grouped_twostep"
        }),
    )
