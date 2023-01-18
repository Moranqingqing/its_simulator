
"""
A simple class to use TrafficEnv as a single agent gym environment
"""
class TrafficEnvSingleAgentGymWrapper:

    def __init__(self, traffic_env):
        if len(traffic_env._agents) != 1:
            raise Exception()

        agent = next(iter(traffic_env._agents.values()))
        self.agent_id = agent.get_id()
        self.traffic_env = traffic_env
        self.action_space = agent.action_space()
        self.observation_space = agent.obs_space()

    def reset(self):
        return self.traffic_env.reset()[self.agent_id]

    def step(self, action):
        rl_actions = {
            self.agent_id: action
        }

        o, r, done, info = self.traffic_env.step(rl_actions)
        return o[self.agent_id], r[self.agent_id], done[self.agent_id] or done["__all__"], info[self.agent_id]