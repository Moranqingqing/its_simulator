from abc import ABC, abstractmethod

from gym.spaces import Tuple
from gym.spaces.dict import Dict

from wolf.world.environments.wolfenv.agents.wolf_agent import WolfAgent

class AgentFactory(ABC):


    @abstractmethod
    def create_agents(self, wolf_env):
        # should return a list of agents and group_agents_params
        raise NotImplementedError


"""
Create a single agent that control multiple connectors. The action/state spaces are joint.
"""
class GlobalAgent(AgentFactory):

    def __init__(self, agent_name):
        AgentFactory.__init__(self)
        self.agent_name = agent_name

    @abstractmethod
    def get_all_connectors_ids(self, wolf_env):
        raise NotImplementedError

    def create_agents(self, wolf_env, **kwargs):
        default_policy = kwargs["default_policy"]
        action_params = kwargs["action_params"]
        obs_params = kwargs["obs_params"]
        reward_params = kwargs["reward_params"]
        if default_policy is not None:
            from wolf.utils.configuration.registry import R
            cls = R.policy_class(default_policy["name"])
            default_policy_cls = cls
            default_policy_params = default_policy["params"]
        else:
            default_policy_cls = None
            default_policy_params = {}

        kernel = wolf_env.get_kernel()
        agents = []
        connectors_ids = self.get_all_connectors_ids(wolf_env)

        action_params = {
            "name": action_params["name"],
            "params": {**action_params["params"],
                       "connectors_ids": connectors_ids}
        }
        obs_params = {
            "name": obs_params["name"],
            "params": {**obs_params["params"],
                       "connectors_ids": connectors_ids}
        }
        reward_params = {
            "name": reward_params["name"],
            "params": {**reward_params["params"],
                       "connectors_ids": connectors_ids}
        }
        tl = WolfAgent.from_string(self.agent_name,
                                   action_params,
                                   obs_params,
                                   reward_params,
                                   kernel=kernel,
                                   default_policy=default_policy_cls,
                                   default_policy_params=default_policy_params)
        agents.append(tl)
        return agents

class GlobalTrafficLightsAgent(GlobalAgent):

    def __init__(self):
        GlobalAgent.__init__(self,"tl")

    def get_all_connectors_ids(self, wolf_env):
        return wolf_env.get_kernel().get_controlled_nodes_ids()



"""
All agents of the environment share the same configuration, ie
same action/state/reward/done connectors.

That doesn't mean all agents will share the same policy. 
Cf multi_agent_settings_factory.py to configure that kind of things
"""
class AllTheSame(AgentFactory):

    def __init__(self, agents_prefix_name):
        AgentFactory.__init__(self)
        self.agents_prefix_name = agents_prefix_name

    @abstractmethod
    def get_all_connectors_ids(self, wolf_env):
        raise NotImplementedError


    def create_agents(self, wolf_env, **kwargs):
        default_policy = kwargs["default_policy"]
        action_params = kwargs["action_params"]
        obs_params = kwargs["obs_params"]
        reward_params = kwargs["reward_params"]
        global_reward = kwargs["global_reward"]

        if default_policy is not None:
            from wolf.utils.configuration.registry import R
            cls = R.policy_class(default_policy["name"])
            default_policy_cls = cls
            default_policy_params = default_policy["params"]
        else:
            default_policy_cls = None
            default_policy_params = {}

        kernel = wolf_env.get_kernel()
        # we link all the traffic lights agents to the node they will controls
        agents = []
        if global_reward:
            reward_params = {
                "name": reward_params["name"],
                "params": {**reward_params["params"],
                           "connectors_ids": kernel.get_controlled_nodes_ids()}
            }

        for connector_id in self.get_all_connectors_ids(wolf_env):
            action_params = {
                "name": action_params["name"],
                "params": {**action_params["params"],
                           "connectors_ids": [connector_id]}
            }
            obs_params = {
                "name": obs_params["name"],
                "params": {**obs_params["params"],
                           "connectors_ids": [connector_id]}
            }
            # print(obs_params)
            if not global_reward:
                reward_params = {
                    "name": reward_params["name"],
                    "params": {**reward_params["params"],
                               "connectors_ids": [connector_id]}
                }
            tl = WolfAgent.from_string("{}_{}".format(self.agents_prefix_name,connector_id),
                                       action_params,
                                       obs_params,
                                       reward_params,
                                       kernel=kernel,
                                       default_policy=default_policy_cls,
                                       default_policy_params=default_policy_params)
            agents.append(tl)

        return agents

class AllTheSameTrafficLights(AllTheSame):
    def __init__(self):
        AllTheSame.__init__(self,"tl")

    def get_all_connectors_ids(self, wolf_env):
        return wolf_env.get_kernel().get_controlled_nodes_ids()

class AllTheSameVehicles(AllTheSame):
    def __init__(self):
        AllTheSame.__init__(self, "veh")

    def create_agents(self, wolf_env, **kwargs):
        """ Function used in initialize stage """
        veh_ids = self.get_all_connectors_ids(wolf_env.get_kernel())
        
        # FIXME: Temp solution for `factory.create_multi_agent_config()` in `WolfEnv.__init__`
        if len(veh_ids) == 0:
            # HACK: this method is only called in the initialization stage
            # We need to bypass the multi_agent_config factory
            veh_ids = ['fake_0']
            
        agents = self.create_agents_by_ids(wolf_env, veh_ids, **kwargs)

        return agents
    
    def create_agents_by_ids(self, wolf_env, veh_ids, **kwargs):
        default_policy = kwargs["default_policy"]
        action_params = kwargs["action_params"]
        obs_params = kwargs["obs_params"]
        reward_params = kwargs["reward_params"]
        global_reward = kwargs["global_reward"]

        if default_policy is not None:
            from wolf.utils.configuration.registry import R
            cls = R.policy_class(default_policy["name"])
            default_policy_cls = cls
            default_policy_params = default_policy["params"]
        else:
            default_policy_cls = None
            default_policy_params = {}

        kernel = wolf_env.get_kernel()
        # we link all the traffic lights agents to the node they will controls
        agents = []
        if global_reward:
            reward_params = {
                "name": reward_params["name"],
                "params": {**reward_params["params"],
                           "connectors_ids": veh_ids}
            }

        for veh_id in veh_ids:
            action_params = {
                "name": action_params["name"],
                "params": {**action_params["params"],
                           "connectors_ids": [veh_id]}
            }
            obs_params = {
                "name": obs_params["name"],
                "params": {**obs_params["params"],
                           "connectors_ids": [veh_id]}
            }
            
            if not global_reward:
                reward_params = {
                    "name": reward_params["name"],
                    "params": {**reward_params["params"],
                               "connectors_ids": [veh_id]}
                }
            
            veh = WolfAgent.from_string("{}_{}".format(self.agents_prefix_name, veh_id),
                                       action_params,
                                       obs_params,
                                       reward_params,
                                       kernel=kernel,
                                       default_policy=default_policy_cls,
                                       default_policy_params=default_policy_params)
            agents.append(veh)
        
        return agents

    # def get_rl_vehicle_ids(self, wolf_kernel):
    #     return wolf_kernel.get_rl_vehicle_ids()

    def get_all_connectors_ids(self, wolf_kernel):
        return wolf_kernel.get_rl_vehicle_ids()