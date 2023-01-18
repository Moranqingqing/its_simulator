"""
Run the CarFollowing environment in simulation
"""
import os

from wolf.world.environments.env_factories import car_following_test
from wolf.config import WOLF_PATH
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams, TrafficLightParams, VehicleParams

from flow.networks.base import Network

vehicles = VehicleParams()
inflow = InFlows()

SCALING = 10
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
flow_rate = 200 * SCALING
AV_FRAC = 0.10

# inflow.add(
#             veh_type="human",
#             edge=7569212471,
#             vehs_per_hour=flow_rate * (1 - AV_FRAC),
#             departLane="random",
#             departSpeed=0)
# inflow.add(
#             veh_type="human",
#             edge=28455502,
#             vehs_per_hour=flow_rate * (1 - AV_FRAC),
#             departLane="random",
#             departSpeed=0)



env_config = {
    'agents_params': {
        'name': 'all_the_same_vehicles_agents',
        'params': {
            'global_reward': False,
            'default_policy': None,
            'action_params': {
                'name': 'VehActionConnector',
                'params': {}
            },
            'obs_params': {
                'name': 'CarFollowingConnector',
                'params': {}
            },
            'reward_params': {
                'name': 'VehRewardConnector',
                'params': {'W2': 5, 'W4': 500,}
            },
        },
    },
    'multi_agent_config_params': {
        'name': 'shared_policy',
        'params': {}
    },
    'group_agents_params': None,
    'sim_params': {
        'restart_instance': True,
        'sim_step': 0.1,
        'render': False,
    },
    'env_state_params': None,
    'action_repeat_params': None,
    'simulator': 'traci',
    'record_flag': False,
    'reward_folder': os.path.join(WOLF_PATH, 'temp@')
}
env = car_following_test(env_config)

env.reset() #env.env_params.horizon

for _ in range(10000):
    env.step({})