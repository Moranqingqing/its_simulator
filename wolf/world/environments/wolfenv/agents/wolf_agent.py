from abc import ABC




class WolfAgent(ABC):
    """
    composition pattern for any type of agent. Can use any kind of action, reward or observation.
    """

    def __init__(self, id, action_connector, observation_connector, reward_connector, done_connector,
                 default_policy=None, default_policy_params={}):
        self._id = id
        self._default_policy = default_policy
        self._default_policy_params = default_policy_params
        self._action_connector = action_connector
        self._observation_connector = observation_connector
        self._reward_connector = reward_connector
        self._done_connector = done_connector
        self._connectors = [action_connector, observation_connector, reward_connector, done_connector]
        self.actions = []
        self.observations = []
        self.rewards = []
        self.dones = []
        for conn in self._connectors:
            conn.attach_agent(self)

    @staticmethod
    def from_string(id,
                    action_params,
                    obs_params,
                    reward_params,
                    kernel,
                    default_policy=None,
                    default_policy_params={}):
        from wolf.utils.configuration.registry import R
        from wolf.world.environments.wolfenv.agents.connectors.done.mock_done_connector import MockDoneConnector
        cls = R.connector_class(action_params["name"])
        action_connector = cls(**action_params["params"], kernel=kernel)
        cls = R.connector_class(obs_params["name"])
        kwargs= dict(action_connector=action_connector) # @parth please implement attach_agent method in TDTSE and retreive action_conn from there, cf TODO in TDTSE
        observation_connector = cls(**obs_params["params"], kernel=kernel, **kwargs)
        cls = R.connector_class(reward_params["name"])
        reward_connector = cls(**reward_params["params"], kernel=kernel)
        done_connector = MockDoneConnector()
        # print(observation_connector)
        return WolfAgent(id, action_connector, observation_connector,
                         reward_connector, done_connector,
                         default_policy, default_policy_params)

    def reset(self):
        for conn in self._connectors:
            conn.reset()
        self.actions.clear()
        self.observations.clear()
        self.rewards.clear()
        self.dones.clear()

    def get_id(self):
        return self._id

    def act(self, action):
        self._action_connector.compute(action)
        self.actions.append(action)

    def observe(self):
        obs = self._observation_connector.compute()
        self.observations.append(obs)
        return obs

    def rewarded(self):
        rew= self._reward_connector.compute()
        self.rewards.append(rew)
        return rew

    def is_done(self):
        done= self._done_connector.compute()
        self.dones.append(done)

    def default_policy(self):
        return self._default_policy

    def default_policy_params(self):
        return self._default_policy_params

    def obs_space(self):
        return self._observation_connector.obs_space()

    def action_space(self):
        return self._action_connector.action_space()

    def get_reward_space(self):
        return self._reward_connector.reward_space()

    def deregister(self, env):
        return