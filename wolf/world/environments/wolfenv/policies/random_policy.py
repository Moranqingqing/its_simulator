from ray.rllib.policy.policy import Policy


class RandomPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self,observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return random actions
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]
