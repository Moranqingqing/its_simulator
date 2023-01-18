from ray.rllib.agents.pg import pg
import ray
from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper
from wolf.world.environments.env_factories import grid_gaussian_master_slaves
from ray.tune.registry import register_env

env_config = {
    "simulator": "traci",
    "sim_params": {
        "restart_instance": True,
        "sim_step": 1,
        "print_warnings": False,
        "render": True,
    },
    "env_state_params": None,
    "group_agents_params": None,
    "multi_agent_config_params": {
        "name": "shared_policy",
        "params": {}
    },

    "agents_params": {
        "name": "all_the_same",
        "params": {
            "global_reward": False,
            "default_policy": None,

            "action_params": {
                "name": "ExtendChangePhaseConnector",
                "params": {},
            },
            "obs_params": {
                "name": "TDTSEConnector",
                "params": {
                    "obs_params": {
                        "num_history": 60,
                        "detector_position": [5, 100],
                    },
                    "phase_channel": True
                }
            },
            "reward_params": {
                "name": "QueueRewardConnector",
                "params": {
                    "stop_speed": 2
                }
            }
        }
    }

}

# create the wolfenv factory (needed for ray)
env_factory = lambda config: grid_gaussian_master_slaves(config, 3, 300)

# register wolfenv factory in Ray
register_env("hello_world_env", env_factory)

# retrieve multi agent config
test_env = env_factory(env_config)
if isinstance(test_env, _GroupAgentsWrapper):
    multi_agent_config = test_env.env.multi_agent_config
else:
    multi_agent_config = test_env.multi_agent_config

# initialise ray
ray.init()

# initialise trainer
trainer = pg.PGTrainer(
    env="hello_world_env",
    config={
        "seed": None,
        "num_workers": 1,
        "multiagent": multi_agent_config,
        "env_config": env_config
    }
)

# train loop
while True:
    x = trainer.train()
    print(x)
