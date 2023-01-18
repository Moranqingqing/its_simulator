from wolf.world.environments.env_factories import grid_gaussian_master_slaves
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND

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
env = grid_gaussian_master_slaves(env_config, 3, 300)

env.reset()
for _ in range(5000):
    env.step({"tl_center0": EXTEND})
