import os
from pathlib import Path

from wolf.world.environments.env_factories import test0

os.environ['SUMO_HOME'] = str(Path.home() / "sumo_binaries" / "bin")
os.environ['PATH'] = "/home/ncarrara/sumo_binaries/bin" + ":" + os.environ["PATH"]
# sim_params = {
#     "restart_instance": False,
#     "sim_step": 1,
#     "print_warnings": False,
#     "render": True,
# }
#
# multi_agent_config_params = {
#     "name": "single_policy",
#     "params": {}
# }
#
# agents_params = {
#     "name": "all_the_same",
#     "params": {
#         "global_reward": False,
#         "default_policy": None,
#
#         "action_params": {
#             "name": "ExtendChangePhaseConnector",
#             "params": {},
#         },
#         "obs_params": {
#             "name": "TDTSEConnector",
#             "params": {
#                 "obs_params": {
#                     "num_history": 60,
#                     "detector_position": [5, 100],
#                 },
#                 "phase_channel": True
#             }
#         },
#         "reward_params": {
#             "name": "QueueRewardConnector",
#             "params": {
#                 "stop_speed": 2
#             }
#         }
#     }
# }
#
# config = {'render': False,
#           'simulator': 'traci',
#           'sim_params': sim_params,
#           'env_state_params': None,
#           'groups_agent_params': None,
#           'multi_agent_config_params': multi_agent_config_params,
#           'agents_params': agents_params}
#
# env = test0(config)
from wolf.world.environments.traffic.traffic_env import TrafficEnv
from wolf.world.environments.traffic.grid_env import SimpleGridEnv

simulator = "traci"

sim_params = {
    "restart_instance": False,
    "sim_step": 1,
    "print_warnings": False,
    "render": True,
}
env_state_params = None

groups_agent_params = None

multi_agent_config_params = {
    "name": "single_policy",
    "params": {}
}

agents_params = {
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

env = TrafficEnv.create_env(
    cls=SimpleGridEnv,
    agents_params=agents_params,
    env_state_params=env_state_params,
    groups_agent_params=groups_agent_params,
    multi_agent_config_params=multi_agent_config_params,
    n=1,
    m=1,
    WE_mu_sigma=(0.2, 1.),
    EW_mu_sigma=(0., 0.),
    NS_mu_sigma=(0., 0.),
    SN_mu_sigma=(0., 0.),
    horizon=50,
    simulator=simulator,
    sim_params=sim_params)