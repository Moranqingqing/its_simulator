from copy import deepcopy

import ray
from gym.spaces import Discrete
from ray import tune
from ray.rllib.agents.pg import pg
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
import gym
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from sow45.utils.configuration.configuration import Configuration
from sow45.world.environments.factories.env_meta_factory import META_FACTORY
from sow45.world.environments.traffic.agents.connectors.action.exchange_change_phase import ExtendChangePhaseConnector
from wolf.world.environments.traffic.agents.connectors.done.mock_done_connector import MockDoneConnector
from sow45.world.environments.traffic.agents.connectors.observation.mock import MockObservationConnector
from sow45.world.environments.traffic.agents.connectors.reward.mock import MockRewardConnector
from sow45.world.environments.traffic.agents.traffic_light import TrafficLight
from sow45.world.environments.traffic.phase import GrGr, rGrG
from sow45.world.environments.traffic.traffic_env import TrafficEnv
from ray.tune import run_experiments
import json

C = Configuration().load("configs/simple_main.json").create_fresh_workspace(force=True)

create_env, gym_name = META_FACTORY.create_factory(**C["env_params"])

def wrapper_create_env(params=None):
    env = create_env(params)
    agents = []

    phases = [GrGr, rGrG]

    for node_id in env.network.traffic_lights.get_properties().keys():
        tl = TrafficLight(
            id="tl_{}".format(node_id),
            action_connector=ExtendChangePhaseConnector(node_id=node_id, phases=phases),
            observation_connector=MockObservationConnector(Discrete(3)),
            reward_connector=MockRewardConnector(Discrete(100)),
            done_connector=MockDoneConnector())
        agents.append(tl)

    env.register_agents(agents)

    return env


# Register as rllib env
register_env(gym_name, wrapper_create_env)

# get any agent as they all share the same act/obs space
test_env = wrapper_create_env()
any_agent = next(iter(test_env.get_agents().values()))
tl_policy = (None, any_agent.obs_space(), any_agent.action_space(), {})

policy_graphs = {
    "tl_policy": tl_policy
}


def policy_mapping_fn(agent_id):
    if "tl" in agent_id:
        return "tl_policy"


alg_run = "PPO"
horizon = flow_params['env'].horizon
n_rollouts = 100

agent_cls = get_agent_class(alg_run)
config = deepcopy(agent_cls._default_config)
config["num_workers"] = 7
config["train_batch_size"] = horizon * n_rollouts
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [32, 32, 32]})
config["use_gae"] = True
config["lambda"] = 0.97
config["kl_target"] = 0.02
config["num_sgd_iter"] = 10
config["horizon"] = horizon

# save the flow params for replay
flow_json = json.dumps(
    flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
config['env_config']['flow_params'] = flow_json
config['env_config']['run'] = alg_run

# multiagent configuration
if policy_graphs is not None:
    print("policies", policy_graphs)
    config['multiagent'].update({'policies': policy_graphs})
if policy_mapping_fn is not None:
    config['multiagent'].update({'policy_mapping_fn': tune.function(policy_mapping_fn)})
# if policies_to_train is not None:
#     config['multiagent'].update({'policies_to_train': policies_to_train})



ray.init(num_cpus=8, object_store_memory=200 * 1024 * 1024)
exp_config = {
    "run": alg_run,
    "env": gym_name,
    "config": {
        **config
    },
    "checkpoint_freq": 20,
    "checkpoint_at_end": True,
    "max_failures": 999,
    "stop": {
        "training_iteration": 1000,
    },
}

# if flags.checkpoint_path is not None:
#     exp_config['restore'] = flags.checkpoint_path
trials = run_experiments({flow_params["exp_tag"]: exp_config})

