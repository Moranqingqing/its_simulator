from wolf.utils.configuration.registry import R
from wolf.world.environments.env_factories import *
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

for stationary in ["stationary"]:
    for demand_symmetry in demand_symmetry_keys:
        for demand_distribution in demand_distribution_keys:
            for layout_symmetry in layout_symmetry_keys:
                params = dict(
                    real_or_fake="fake",
                    demand_distribution=demand_distribution,
                    demand_symmetry=demand_symmetry,
                    stationary=stationary,
                    layout_symmetry=layout_symmetry,
                    n=1,
                    m=1)

                print("params:", params)

                env = benchmark_env(
                    env_config=env_config,
                    **params
                )


env = benchmark_env(env_config=env_config, real_or_fake="real")

