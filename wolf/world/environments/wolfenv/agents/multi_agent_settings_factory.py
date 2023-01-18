from abc import ABC, abstractmethod

from gym.spaces import Tuple
from gym.spaces.dict import Dict

from wolf.world.environments.wolfenv.agents.wolf_agent import WolfAgent


class MultiAgentConfigFactory(ABC):
    @staticmethod
    @abstractmethod
    def create_multi_agent_config(agents, **kwargs):
        raise NotImplementedError


class SharedPolicy(MultiAgentConfigFactory):
    """
    All agents share the same weights.
    """

    @staticmethod
    def create_multi_agent_config(agents, **kwargs):
        any_agent = next(iter(agents.values()))
        policy_graphs = {
            "unique_policy": (any_agent.default_policy(),
                              any_agent.obs_space(),
                              any_agent.action_space(),
                              any_agent.default_policy_params())
        }

        def policy_mapping_fn(agent_id):
            # if "tl" in agent_id:
            return "unique_policy"

        return {"policies": policy_graphs, "policy_mapping_fn": policy_mapping_fn}


class IndependentPolicy(MultiAgentConfigFactory):
    """
    For heterogeneous agents.
    """

    @staticmethod
    def create_multi_agent_config(agents, **kwargs):
        policy_graphs = {}
        for agent in agents.values():
            policy_graphs[agent.get_id()] = (
                agent.default_policy(),
                agent.obs_space(),
                agent.action_space(),
                agent.default_policy_params()
            )

        def policy_mapping_fn(agent_id):
            return agent_id

        return {"policies": policy_graphs, "policy_mapping_fn": policy_mapping_fn}


class GroupAgentsParamsFactory(ABC):
    @staticmethod
    @abstractmethod
    def group_agents_params(agents, **kwargs):
        raise NotImplementedError


class SingleGroup(GroupAgentsParamsFactory):
    @staticmethod
    def group_agents_params(env, **kwargs):
        agents = env.get_agents().items()
        state_env_space = env.observation_space()
        action_spaces = [agent.action_space() for i, agent in agents]
        obs_spaces = []
        for i, agent in agents:  # recontruct such that there is an agent obs and an wolfenv state
            obs_spaces.append(
                Dict(obs=agent.obs_space(),
                     state=state_env_space))
        group_agents_params = {
            "groups": {
                "group_1": [agent.get_id() for i, agent in agents]
            },
            "obs_space": Tuple(obs_spaces),
            "act_space": Tuple(action_spaces)
        }

        return group_agents_params
