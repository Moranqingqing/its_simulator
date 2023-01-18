import argparse
import logging
import os
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND
from wolf.world.environments.wolfenv.car_following_env import BasicCFMController, CarFollowingEnv, CustomizedGippsController, CustomizedIDMController, CustomizedBCMController, CustomizedDummyController
from wolf.world.environments.wolfenv.wolf_env import WolfEnv
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams, TrafficLightParams, VehicleParams
import flow.config as config

import random

from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController
import gym
import numpy as np
import torch
import pandas as pd
import pickle as pkl

from flow.networks.i210_subnetwork import I210SubNetwork
from flow.envs.multiagent.i210 import I210MultiEnv, ADDITIONAL_ENV_PARAMS

from ddpg import DDPG

# Create logger
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def constrain_action(state, accel, epsilon=0.5):
    veh_speed, rel_speed, distance_headway, follow_rel_speed, follow_distance_headway, speed_limit, prev_accel = state.cpu().numpy()
    if distance_headway<20 and accel>0:
        accel=np.clip(-2*accel, -3, 3)
    elif veh_speed >= speed_limit:
        if veh_speed > speed_limit + epsilon:
            accel = np.clip([speed_limit - veh_speed], -3, 3)
        else:
            accel = np.clip(accel, -3, 0)
    elif distance_headway > 200:
        if veh_speed <= speed_limit - epsilon:
            accel = np.clip([speed_limit - veh_speed], -3, 3)
    return accel

def batch_constrain_action(state, accels, epsilon=0.5):
    veh_speed, rel_speed, distance_headway, follow_rel_speed, follow_distance_headway, speed_limit, prev_accel = np.split(state.cpu().numpy(), 7, axis=1)

    # condition_1 = (distance_headway < 50) & (accels > 0)
    condition_2 = (veh_speed > speed_limit + epsilon)
    condition_3 = (veh_speed >= speed_limit) & (veh_speed <= speed_limit + epsilon)
    condition_4 = (distance_headway > 200) & (veh_speed <= speed_limit - epsilon)

    # accels = np.clip(np.where(condition_1, -2*accels, accels), -3, 3)
    accels = np.where(condition_2, np.clip(speed_limit-veh_speed, -3, 0), accels)
    accels = np.where(~condition_2 & condition_3, np.clip(accels, -3, 0), accels)
    accels = np.where(~condition_2 & ~condition_3 & condition_4, np.clip(speed_limit - veh_speed, 0, 3), accels)

    return accels

# Parse given arguments
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=0, help="Random seed")
parser.add_argument("--save_dir", default="./saved_modelsnew/", help="Dir. path to load a model")
parser.add_argument("--load_model", help="Path to load a model")
parser.add_argument("--episodes", default=1, help="Num. of test episodes")
parser.add_argument("--network", default="i20", choices=['i20'])
parser.add_argument("--speed-limit", default=None, type=int)
parser.add_argument("--speed-reward", default=True, action='store_true')
parser.add_argument("--no-speed-reward", dest="speed_reward", action='store_false')
parser.add_argument("--controller", default="rl", choices=["rl", "idm", "gipps", "bcm", "cfm"])
parser.add_argument("--lv", default=True, action="store_true", help="Enable leading vehicle [Enabled by default]")
parser.add_argument("--no-lv", dest="lv", action="store_false", help="Disable leading vehicle")
parser.add_argument("--constrain", default=True, action="store_true", help="Enable constrain [Enabled by default]")
parser.add_argument("--no-constrain", dest="constrain", action="store_false", help="Disable constrain")
parser.add_argument("--perturbation-test", default=False, action="store_true", help="Test perturbation, this will change all initial speed to 25m/s")
parser.add_argument("--perturbation-rl-num", default=8, type=int, help="Set the number of rl vehicles in the perturbation test")
parser.add_argument("--initial-speed", default=15, type=float, help="Set the initial speed of vehicles in the perturbation test")
parser.add_argument("--render", default=False, action="store_true", help="Enable the render of the simulator")
parser.add_argument("--USE_INFLOWS", default=False, action="store_true", help="Enable the inflow of the simulator")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


vehicles = VehicleParams()

random.seed(0)
np.random.seed(0)


if args.controller == 'rl':
    controller = (RLController, {})
elif args.controller == 'gipps':
    controller = (CustomizedGippsController, {"acc": 3})
elif args.controller == 'idm':
    controller = (CustomizedIDMController, {"a": 3, "T": 1.26 if not args.perturbation_test else 0.8})
elif args.controller == 'bcm':
    controller = (CustomizedBCMController, {"k_c": 0})
elif args.controller == 'cfm':
    controller = (BasicCFMController, {"k_c": 0})

# SET UP PARAMETERS FOR THE ENVIRONMENT
additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
additional_env_params.update({
    'max_accel': 1,
    'max_decel': 1,
    "lane_change_duration": 5,
    # configure the observation space. Look at the I210MultiEnv class for more info.
    'lead_obs': True,
    "disable_tb": True,  ##??
    "disable_ramp_metering": True, ##??
    "add_rl_if_exit": False
})


# CREATE VEHICLE TYPES AND INFLOWS
# no vehicles in the network


vehicles = VehicleParams()
vehicles.add(
    veh_id="human_follower",
    # acceleration_controller=(RLController, {}),
    # lane_change_controller=(SimLaneChangeController, {}),
    acceleration_controller=(CustomizedIDMController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="all_checks",
        accel=3,
        decel=3,
        max_speed=30
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
    num_vehicles=1,
    initial_speed=0 if not args.perturbation_test else args.initial_speed)

vehicles.add(
    veh_id="followerstopper",
    acceleration_controller=controller,
    # lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="aggressive",
        accel=3,
        decel=3,
        # max_speed=70,
        # speed_dev=0
    ),
    # lane_change_params=SumoLaneChangeParams(
    #     lane_change_mode=0,
    # ),
    num_vehicles=10,
    initial_speed=0 if not args.perturbation_test else args.initial_speed) # set initial speed

if args.lv:
    vehicles.add(
        veh_id="human",
        # acceleration_controller=(RLController, {}),
        lane_change_controller=(SimLaneChangeController, {}),
        acceleration_controller=(CustomizedDummyController, {'v_des': args.initial_speed}) if args.perturbation_test else (CustomizedIDMController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="aggressive" if args.perturbation_test else "all_checks",
            accel=3,
            decel=3,
            max_speed=30
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,
        ),
        num_vehicles=1,
        initial_speed=0 if not args.perturbation_test else args.initial_speed)



inflow = InFlows()
# main highway
PENETRATION_RATE=10
pen_rate = PENETRATION_RATE / 100
assert pen_rate < 1.0, "your penetration rate is over 100%"
assert pen_rate > 0.0, "your penetration rate should be above zero"
inflow.add(
    veh_type="human",
    edge="119257914",
    vehs_per_hour=8378 * pen_rate,
    # probability=1.0,
    departLane="random",
    departSpeed=20)

inflow.add(
    veh_type="followerstopper",
    edge="119257914",
    vehs_per_hour=int(8378 * pen_rate),
    # probability=1.0,
    departLane="random",
    departSpeed=20)

traffic_lights = TrafficLightParams()
NET_TEMPLATE = os.path.join(
    config.PROJECT_PATH,
    "examples/exp_configs/templates/sumo/test2.net.xml")

net=NetParams(
    inflows=inflow,
    template=NET_TEMPLATE,
    additional_params={
        "on_ramp": False,
        "ghost_edge": False
    })

network = I210SubNetwork(
    name='I210SubNetwork',
    vehicles=vehicles,
    net_params=net,
    initial_config=InitialConfig(edges_distribution=['119257914']),
    # traffic_lights=traffic_lights
)

env_config = {
    "simulator": "traci",

    "sim_params": SumoParams(
        sim_step=0.1,
        render=args.render,
        print_warnings=False,
        restart_instance=True,
        emission_path='/tmp/yifei'
    ),

    "env_params": EnvParams(
        warmup_steps=200,
        sims_per_step=1,
        horizon=3000,
        additional_params=additional_env_params,
    ),

    "network": network,

    "env_state_params": None,
    "group_agents_params": None,
    "multi_agent_config_params": None,
    "agents_params": {
        "name": "all_the_same_vehicles_agents",
        "params": {
            "global_reward": False,
            "default_policy": None,

            "action_params": {
                "name": "VehActionConnector",
                "params": {},
            },
            "obs_params": {
                "name": "BCMObsConnector",
                "params": {}
            },
            "reward_params": {
                "name": "BCMVehRewardConnector",
                "params": {
                    "W1": [1, 1],
                    "W2": [1, 1],
                    "W3": [1, 1],
                    "W4": [1, 1]
                }
            }
        }
    }
}

env = WolfEnv.create_env(
    cls=CarFollowingEnv,
    network=network,
    env_params=env_config['env_params'],
    sim_params=env_config['sim_params'],
    agents_params=env_config["agents_params"],
    env_state_params=env_config["env_state_params"],
    group_agents_params=env_config["group_agents_params"],
    multi_agent_config_params=env_config["multi_agent_config_params"],
    simulator=env_config["simulator"],
    action_repeat_params=env_config.get("action_repeat_params", None),
)


# Enable the environment recording
env.record_train = False
env.record_eval = True
env.trial_id = None



if __name__ == "__main__":

    logger.info("Using device: {}".format(device))

    random.seed(0)
    np.random.seed(0)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = args.save_dir


    # agent = DDPG(gamma,
    #              tau,
    #              hidden_size,
    #              env.observation_space.shape[0],
    #              env.action_space,
    #              checkpoint_dir=checkpoint_dir
    #              )
    agent = DDPG(0.99,
                    0.001,
                    (400, 300),
                    7, ##state space
                    np.array([1,]), ##action space
                    checkpoint_dir=checkpoint_dir
                    )
    if args.controller == 'rl':
        agent.load_checkpoint(args.load_model, training=False)

    # Load the agents parameters
    agent.set_eval()
    # controller='DDPG'


    for _ in range(args.episodes):
        step = 0
        returns = list()
        states=[]
        actions=[]
        lv_speed=[]
        lv_acc=[]
        state=env.reset()
        agent_names = list(state.keys())
        batch_state = np.array([state[agent_name] for agent_name in agent_names])
        # print('state',state)
        if args.controller == 'rl':
            state = torch.Tensor(batch_state).to(device)
        speed_avg=[]
        while True:
            if args.controller == 'rl':
                states.append(state.cpu().numpy())

                action = agent.calc_action(state, action_noise=None)

                speed=0
                for j in range(len(state)):
                    speed = speed+ state[j][0]
                average_speed=speed/len(state)
                speed_avg.append(average_speed)

                actions.append(action)

                accels = action.cpu().numpy()
                if args.constrain:
                    accels = batch_constrain_action(state, accels)
                
                action_dir = dict((agent_name, accel) for agent_name, accel in zip(agent_names, accels))

                next_state, reward, done_dict, _ = env.step(action_dir)
                done=done_dict.get('__all__')
                if done:
                    next_state = None
                else:
                    agent_names = list(next_state.keys())
                    for agent_name in done_dict:
                        if done_dict[agent_name] and agent_name != '__all__':
                            agent_names.remove(agent_name)
                    batch_state = np.array([next_state[agent_name] for agent_name in agent_names])
                    state = torch.Tensor(batch_state).to(device)

            else:
                next_state, reward, done, _ = env.step({})
                done = done.get('__all__')

            step += 1
            if done:
                break

        env_record = env.history_record
        env_global_metric = env.global_metric
        # print('average speed',np.mean(speed_avg))


        # Convert it to pandas DataFrame
        stats = pd.DataFrame(env_record)

        # Some annotation for filename
        save_dir = os.path.dirname(args.load_model)
        model_name = os.path.basename(args.load_model)
        curr_steps = ''
        if args.controller == 'rl':
            curr_steps = '_'+''.join(filter(str.isdigit, model_name))
        lv_str = 'lv' if args.lv else 'no-lv'
        speed_limit_str = args.speed_limit if args.speed_limit else 'dynamic'
        constraint_str = 'constrain' if args.constrain else 'no-constrain'
        with open(os.path.join(save_dir, 
            f'ddpg_test{curr_steps}_{args.controller}_{lv_str}_{args.network}_speed_{speed_limit_str}_{constraint_str}.pkl'), 'wb') as f:
            pkl.dump(stats, f)
        with open(os.path.join(save_dir, 
            f'ddpg_test{curr_steps}_{args.controller}_{lv_str}_{args.network}_speed_{speed_limit_str}_{constraint_str}_global_metric.pkl'), 'wb') as f:
            pkl.dump(env_global_metric, f)

    mean = np.mean(returns)
    variance = np.var(returns)
    logger.info("Score (on 100 episodes): {} +/- {}".format(mean, variance))