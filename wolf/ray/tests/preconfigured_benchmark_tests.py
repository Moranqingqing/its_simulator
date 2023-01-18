from wolf.utils.configuration.registry import R
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND

restart_instance = False

env_config = {
    "simulator": "traci",

    "sim_params": {
        "restart_instance": restart_instance,
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

env = R.env_factory("benchmark_0")(env_config)

env.reset()
from time import sleep
for _ in range(1000):
    env.step({"tl_center0": EXTEND})
    # env.step({"main_center": EXTEND})
    sleep(0.05)
