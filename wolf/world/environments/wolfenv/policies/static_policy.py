from ray.rllib.policy.policy import Policy

from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import (
    CHANGE,
    EXTEND,
)


class StaticMinPolicy(Policy):
    """
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.w = 1.0

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        timestep=None,
        **kwargs
    ):
        return [CHANGE for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]


class StaticMaxPolicy(Policy):
    """
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.w = 1.0

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        timestep=None,
        **kwargs
    ):
        return [EXTEND for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}  # return stats

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]


class GlobalGreenWavePolicy(Policy):
    # Made for test0_1 environment.
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.w = 1.0

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        timestep=None,
        **kwargs
    ):
        # TODO (parth): remove hard-coding, and make generalizable using platoon_time and no_platoon_time.
        current_timestep = episodes[0].length + 1
        rem1 = (current_timestep - 10) % 15
        rem2 = (current_timestep - 20) % 15

        # 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11.
        # above are binary representations which are used by the action connector.
        if rem1 == 0 and current_timestep > 10:
            return [2 for _ in obs_batch], [], {}
        elif rem2 == 0 and current_timestep > 20:
            return [1 for _ in obs_batch], [], {}
        else:
            return [0 for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        return {}  # return stats

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]
