"""
Run the CarFollowing environment in simulation
"""
import os

from wolf.world.environments.env_factories import car_following_qew
from wolf.config import WOLF_PATH

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
        'restart_instance': False,
        'sim_step': 0.1,
        'horizon': 5000,
        'warmup_steps': 0,
        'stats_collection_interval': (0, 0, 10),
        'render': False,
    },
    'env_state_params': None,
    'action_repeat_params': None,
    'simulator': 'aimsun',
    'record_flag': False,
    'reward_folder': os.path.join(WOLF_PATH, 'temp@')
}

if env_config['simulator'] == 'aimsun':
    env_config['sim_params'].update({
        'restart_instance': False,
        'replication': 4321,
    })

env = car_following_qew(env_config)

obs = env.reset()
for _ in range(env.sim_params.horizon):
    obs, rew, info, done = env.step({})
    print(obs)
