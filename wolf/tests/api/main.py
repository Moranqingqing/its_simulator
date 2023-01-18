# TODO: Convert to Pytest
import unittest

from wolf.ray.main import runs
from wolf.world.environments.ctm.ctm_env import EXTEND
from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv
from wolf.world.environments.wolfenv.wolf_env import WolfEnv

simulator = "traci"

sim_params = {
    "restart_instance": True,
    "sim_step": 1,
    "print_warnings": False,
    "render": False,
}

env_state_params = None

group_agents_params = None

multi_agent_config_params = {
    "name": "shared_policy",
    "params": {}
}
inflows_params = {"WE": (0.2, 1.),
                  "EW": (0., 0.),
                  "NS": (0., 0.),
                  "SN": (0., 0.)}
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
                    "num_history": 2,
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
detectors_params = {'positions': [-5, -100], 'frequency': 100}


class TestAPIs(unittest.TestCase):

    def test_functional_api(self):
        horizon = 50
        env = WolfEnv.create_env(
            cls=SimpleGridEnv,
            agents_params=agents_params,
            env_state_params=env_state_params,
            group_agents_params=group_agents_params,
            multi_agent_config_params=multi_agent_config_params,
            n=1,
            m=1,
            inflow_params=inflows_params
            ,
            horizon=horizon,
            detector_params=detectors_params,
            simulator=simulator,
            sim_params=sim_params)

        for _ in range(2):
            env.reset()
            for _ in range(horizon):
                env.step({"tl_center0": EXTEND})

    # def test_framework_api(self):
    #     runs(config_file_path="global_agent.yaml",override_config_file_path=None)

if __name__ == '__main__':
    unittest.main()
