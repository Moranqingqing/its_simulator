from abc import ABC, abstractmethod
from collections import OrderedDict
from flow.envs.multiagent import MultiEnv
import logging
import numpy as np

from gym.spaces import Box


class EnvPostprocessing(ABC):
    @abstractmethod
    def compute(self, traffic_env):
        raise NotImplementedError


class WolfEnv(MultiEnv):

    @staticmethod
    def create_env(cls, *args, **kwargs):
        env = cls(*args, **kwargs)
        if env.group_agents_params is not None:
            env = env.with_agent_groups(**env.group_agents_params)
        return env

    def __init__(
            self,
            agents_params,
            group_agents_params,
            multi_agent_config_params,
            env_params,
            sim_params,
            network,
            tl_params,
            inflows_generator=None,
            lane_arrival_sched=None,
            env_state_params=None,
            controlled_nodes=None,
            action_repeat_params=None,
            simulator='traci'
    ):
        """
        Do not use this directly. Use create_env static function instead.

        :param agents_params:
        :param group_agents_params:
            you can discard this if you don't use Ray. Only Ray's Qmix use this feature.
        :param multi_agent_config_params:
            you can discard this if you don't use RAY. It setups the policy graph and mapping for Ray's RL algorithms.
        :param env_params:
        :param sim_params:
        :param network:
        :param tl_params:
        :param inflows_generator:
        :param lane_arrival_sched:
            schedule of lane arrivals by timestep
        :param env_state_params:
            environment state. Agents are not supposed to know this state. Agents will receive local observations.
        :param simulator:
        :param gym_env: classic single agent gym interface
        """
        super().__init__(env_params, sim_params, network, simulator)

        self.logger = logging.getLogger(__name__)
        self.logger.info("[WolfEnv] A new WolfEnv is created")
        self._agents = None
        self._inflows_generator = inflows_generator

        if action_repeat_params:
            self._action_repeat = True
            self._action_repeat_type = action_repeat_params['type']
            self._repeat_count = action_repeat_params['params']['amount']
        else:
            self._action_repeat = False

        # we create our custom kernel (it uses flow kernel and has extra functionalities)
        if simulator == 'traci':
            from wolf.world.environments.wolfenv.kernels.traci_wolf_kernel import TraciWolfKernel
            self.kernel = TraciWolfKernel(self.k, self.network, sim_params, tl_params, controlled_nodes)
        elif simulator == 'aimsun':
            from wolf.world.environments.wolfenv.kernels.aimsun_wolf_kernel import AimsunWolfKernel
            self.kernel = AimsunWolfKernel(self.k, self.network, sim_params, tl_params, controlled_nodes)
        else:
            raise RunTimeError(f'Simulator type "{simulator}" is not valid.')

        from wolf.utils.configuration.registry import R
        factory = R.agent_factory(agents_params["name"])
        agents = factory.create_agents(self, **agents_params["params"])
        self.register_agents(agents)

        if env_state_params is not None:
            cls = R.true_state_class(env_state_params["name"])
            self.env_state_conn = cls(**env_state_params["params"], env=self)
            # self.observation_space = self.env_state_conn.obs_space()
        else:
            self.env_state_conn = None

        if group_agents_params is not None:
            factory = R.group_agents_params_factory(group_agents_params["name"])
            self.group_agents_params = factory.group_agents_params(self, **group_agents_params["params"])
            # self.action_space = self.group_agents_params["act_space"]
        else:
            self.group_agents_params = None

        if multi_agent_config_params is not None:
            factory = R.multi_agent_config_factory(multi_agent_config_params["name"])
            self.multi_agent_config = factory.create_multi_agent_config(self._agents,
                                                                        **multi_agent_config_params["params"])
        else:
            self.multi_agent_config = {}

        if lane_arrival_sched is not None:
            self.set_lane_arrival_sched(lane_arrival_sched)


    def reset(self, new_inflow_rate=None, perform_extra_work=None):
        self.logger.debug("Resetting Env")

        if self._inflows_generator is not None:
            self.logger.debug("Resetting Inflow")
            inflow = self._inflows_generator()
            self.network.net_params.inflows = inflow

        def extra_work():
            self.logger.debug("Performing extra work in flow reset function")

            for id, agent in self.get_agents().items():
                agent.reset()

        if perform_extra_work is None:
            perform_extra_work = extra_work

        obs = super().reset(new_inflow_rate, perform_extra_work=perform_extra_work)

        return obs

    def register_agents(self, agents):
        for agent in agents:
            self.register_agent(agent)

    def register_agent(self, agent):
        if self._agents is None:
            self._agents = OrderedDict()
        if agent.get_id() in self._agents:
            raise RuntimeError(
                "Agent {} is already registered in Environment {}".format(agent.get_id(), self.get_id()))
        self._agents[agent.get_id()] = agent

    def deregister_agent(self, id):
        self._agents[id].deregister(self)
        del self._agents[id]

    def set_lane_arrival_sched(self, lane_arrival_sched):
        """
        Set a schedule of vehicle arrivals by lane to the environment,
        and create an iterable that will be used by the MultiEnv class
        in Flow to generate the vehicles

        Parameter
        ---------
            lane_arrival_sched : tuple of lists of 2-tuples (str, str)
                Tuple of lists of form [(lane_id, veh_id)], indexed by simultation step
        """
        self.network.net_params.lane_arrival_sched = lane_arrival_sched
        self.network.net_params.lane_arrivals_iter = iter(self.network.net_params.lane_arrival_sched)
        self.network.net_params.arrivals_remaining = True

    # @override (from flow.core.envs.wolfenv)
    def _apply_rl_actions(self, rl_actions):
        for i, action in rl_actions.items():
            # removed from Flow code
            # clipping action if box
            if isinstance(self._agents[i].action_space, Box):
                action = np.clip(
                    action,
                    a_min=self.action_space.low,
                    a_max=self.action_space.high)
            self._agents[i].act(action)

    # @override (from flow.core.envs.wolfenv)
    def compute_reward(self, rl_actions, **kwargs):

        if rl_actions is None:  # in case of warmup executions
            return {}
        rews = {}
        for i, action in rl_actions.items():
            rews[i] = self._agents[i].rewarded()
        self._last_rew = rews.copy()
        return rews

    # @override (from flow.core.envs.wolfenv)
    def additional_command(self):
        self.logger.debug("Here goes the additional commands. Called after apply_rl_actions")

    # @override (from flow.core.envs.wolfenv)
    def get_state(self):
        obs = {}
        if self.env_state_conn is not None:
            for i, agent in self._agents.items():
                obs[i] = {}
                obs[i]["obs"] = agent.observe()
            self.state = self.env_state_conn.compute()
            for i, agent in self._agents.items():
                obs[i]["state"] = self.state
        else:
            for i, agent in self._agents.items():
                obs[i] = agent.observe()
        return obs

    def get_agents(self):
        if self._agents is not None:
            return self._agents
        else:
            raise RuntimeError("Agents have not been initialized")

    def get_id(self):
        return self._id

    def get_kernel(self):
        return self.kernel
