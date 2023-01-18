from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND

from wolf.world.environments.wolfenv.wolf_env import WolfEnv
from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

simulator = "traci"
sim_params = {
    "restart_instance": True,
    "sim_step": 1,
    "print_warnings": False,
    "render": True,
}
env_state_params = None
group_agents_params = None
multi_agent_config_params = {
    "name": "shared_policy",
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

horizon = 500

env = WolfEnv.create_env(
    cls=SimpleGridEnv,
    agents_params=agents_params,
    env_state_params=env_state_params,
    group_agents_params=group_agents_params,
    multi_agent_config_params=multi_agent_config_params,
    n=1,
    m=1,
    inflow_params={"WE": (0.2, 1.),
                   "EW": (0., 0.),
                   "NS": (0., 0.),
                   "SN": (0., 0.)}
    ,
    horizon=horizon,
    detector_params={'positions': [-5, -100], 'frequency': 100},
    simulator=simulator,
    sim_params=sim_params)

for _ in range(5):
    print("--------------------------------------")
    env.reset()
    for _ in range(horizon):

        env.step({"tl_center0": EXTEND})
