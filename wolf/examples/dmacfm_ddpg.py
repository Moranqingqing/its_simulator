from numpy.lib.function_base import average
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND
from wolf.world.environments.wolfenv.car_following_env import CarFollowingEnv, CarFollowingNetwork, CarFollowingStraightNetwork, RLVehRouter
from wolf.world.environments.wolfenv.wolf_env import WolfEnv
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams, TrafficLightParams, VehicleParams
from wolf.world.environments.wolfenv.car_following_env import CarFollowingEnv, CarFollowingNetwork, CarFollowingStraightNetwork, CustomizedGippsController, CustomizedIDMController


from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController, IDMController

import numpy as np

##import DDPG settings
import argparse
import logging
import os
import random
import time
import yaml
import re

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
# from wrappers.normalized_actions import NormalizedActions
##

# Generate time string
named_tuple = time.localtime()
time_string = time.strftime("%m-%d_%H:%M:%S", named_tuple)
unix_timestamp = int(time.time())
# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse given arguments
# gamma, tau, hidden_size, replay_size, batch_size, hidden_size are taken from the original paper
parser = argparse.ArgumentParser()

parser.add_argument("--render_train", default=False, type=bool,
                    help="Render the training steps (default: False)")
parser.add_argument("--render_eval", default=False, type=bool,
                    help="Render the evaluation steps (default: False)")
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default=f"./tmp/car_follow_{unix_timestamp}_{time_string}/",
                    help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed (default: 0)")
parser.add_argument("--lv", default=True, action="store_true", help="Enable leading vehicle [Enabled by default]")

parser.add_argument("--timesteps", default=1e5, type=int,
                    help="Num. of total timesteps of training (default: 1e6)") ##1e6
parser.add_argument("--batch_size", default=64, type=int,
                    help="Batch size (default: 64; OpenAI: 128)")
parser.add_argument("--replay_size", default=1e6, type=int,
                    help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001,
                    help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--noise_stddev", default=0.002, type=int,
                    help="Standard deviation of the OU-Noise (default: 0.2)")
parser.add_argument("--hidden_size", nargs=2, default=[400, 300], type=tuple,
                    help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")
parser.add_argument("--n_test_cycles", default=3, type=int,
                    help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
parser.add_argument("--constrain", default=False, action="store_true", help="Enable constrain [Enabled by default]")
parser.add_argument("--no-constrain", dest="constrain", action="store_false", help="Disable constrain")
parser.add_argument("--constrain-type", default=1, type=int, choices=[1, 2, 3, 4, 5], help="Define the constraint type")
parser.add_argument("--speed-reward-weight", type=float, default=0., help="Weight for the speed limit reward component")
parser.add_argument("--speed-reward", default=False, action='store_true')
parser.add_argument("--no-speed-reward", dest="speed_reward", action='store_false')
parser.add_argument("--speed-limit", default=None, type=int)
parser.add_argument("--cmdp", default=False, action='store_true')
parser.add_argument("--episodes", default=200, help="Num. of test episodes")
parser.add_argument("--save_freq", default=10, help="how many episodes to save model")
parser.add_argument("--f-eff-type", default=1, type=int, choices=[1, 2, 3], 
                    help="Choose the type of f_eff, (1: maximum at 1.26s, 2: max at 0.8s, 3: max at 0s)")
parser.add_argument("--follow-ttc", default=True, action="store_true", help="Add follower ttc safety function in the reward function")
parser.add_argument("--no-follow-ttc", dest="follow_ttc", action="store_false", help="Disable follower ttc safety in the reward function")

args = parser.parse_args()



# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))


def constrain_action1(state, accel, epsilon=0.5):
    """ The constraint I think it's good but it doesn't work well
    regarding with the final performance 
    """

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

def constrain_action2(state, accel, epsilon=0.5):
    """ Another constraint implementation which doesn't check 
    overspeeding when the vehicle doesn't have the leading vehicle 
    """
    veh_speed, rel_speed, distance_headway, follow_rel_speed, follow_distance_headway, speed_limit, prev_accel = state.cpu().numpy()
    if distance_headway<20 and accel>0:
        accel=np.clip(-2*accel, -3, 3)
    elif distance_headway > 200:
        if veh_speed <= speed_limit - epsilon:
            accel = np.clip([speed_limit - veh_speed], -3, 3)
    elif veh_speed >= speed_limit + epsilon:
        if veh_speed > speed_limit + epsilon:
            accel = np.clip([speed_limit - veh_speed], -3, 3)
    return accel

def constrain_action3(state, accel, epsilon=0.5):
    """ Another constraint implementation which doesn't check 
    overspeeding when the vehicle doesn't have the leading vehicle 
    """
    print(state)
    veh_speed, rel_speed, distance_headway, follow_rel_speed, follow_distance_headway, speed_limit, prev_accel = state.cpu().numpy()
    if distance_headway<20 and accel>0:
        accel=np.clip(-2*accel, -3, 3)
    elif distance_headway > 200:
        if veh_speed <= speed_limit - epsilon:
            accel = np.clip([speed_limit - veh_speed], -3, 3)
    elif veh_speed >= speed_limit:
        if veh_speed > speed_limit + epsilon:
            accel = np.clip([speed_limit - veh_speed], -3, 3)
        else:
            accel = np.clip(accel, -3, 0)
    return accel

def constrain_action4(state, accel, epsilon=0.5):
    """ Another constraint implementation which doesn't check 
    overspeeding when the vehicle doesn't have the leading vehicle 
    """
    veh_speed, rel_speed, distance_headway, follow_rel_speed, follow_distance_headway, speed_limit, prev_accel = state.cpu().numpy()
    if distance_headway<20 and accel>0:
        accel=np.clip(-2*accel, -3, 3)
    elif distance_headway > 200:
        if veh_speed <= speed_limit - epsilon:
            accel = np.clip([speed_limit - veh_speed], -3, 3)
        elif veh_speed <= speed_limit:
            accel = np.clip(accel, 0, 3)
    elif veh_speed >= speed_limit + epsilon:
        if veh_speed > speed_limit + epsilon:
            accel = np.clip([speed_limit - veh_speed], -3, 3)
    return accel

def constrain_action5(state, accel, epsilon=0.5):
    """ This constraint almost check everything but it seems doesn't have
    a good performance 
    """
    veh_speed, rel_speed, distance_headway, speed_limit, prev_accel = state.cpu().numpy()
    if distance_headway<20 and accel>0:
        accel=np.clip(-2*accel, -3, 3)
    elif veh_speed >= speed_limit + epsilon:
        accel = np.clip([speed_limit - veh_speed], -3, 3)
    elif veh_speed >= speed_limit:
        accel = np.clip(accel, -3, 0)
    elif distance_headway > 200:
        if veh_speed <= speed_limit - epsilon:
            accel = np.clip([speed_limit - veh_speed], -3, 3)
        elif veh_speed <= speed_limit:
            accel = np.clip(accel, 0, 3)
    return accel


constrain_action = [constrain_action1, constrain_action2, constrain_action3,
 constrain_action4, constrain_action5][args.constrain_type-1]


# time horizon of a single rollout
HORIZON = 3000
# number of parallel workers
N_CPUS = 2
# number of rollouts per training iteration
N_ROLLOUTS = N_CPUS * 4

SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.10
reward_threshold=0



# Random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)



vehicles = VehicleParams()



vehicles.add(
    veh_id="human_follower",
    # acceleration_controller=(RLController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    acceleration_controller=(CustomizedIDMController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="all_checks",
        accel=3,
        decel=3,
        max_speed=30
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=1)


vehicles.add(
    veh_id="followerstopper",
    acceleration_controller=(RLController, {}),
    # lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(RLVehRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="aggressive",
    ),
    # lane_change_params=SumoLaneChangeParams(
    #     lane_change_mode=0,
    # ),
    num_vehicles=5 * SCALING)

if args.lv:
    vehicles.add(
        veh_id="human",
        # acceleration_controller=(RLController, {}),
        lane_change_controller=(SimLaneChangeController, {}),
        acceleration_controller=(CustomizedIDMController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="all_checks",
            accel=3,
            decel=3,
            max_speed=30
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,
        ),
        num_vehicles=1)


vehicles2 = VehicleParams()

vehicles2.add(
    veh_id="followerstopper",
    acceleration_controller=(RLController, {}),
    # lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(RLVehRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="aggressive",
    ),
    # lane_change_params=SumoLaneChangeParams(
    #     lane_change_mode=0,
    # ),
    num_vehicles=1 * SCALING)



controlled_segments = [("1", 1, False), ("2", 2, True), ("3", 2, True),
                       ("4", 2, True), ("5", 1, False)]
num_observed_segments = [("1", 1), ("2", 3), ("3", 3), ("4", 3), ("5", 1)]
additional_env_params = {
    # For ADDITIONAL_ENV_PARAMS
    "max_accel": 3,
    "max_decel": 3,
    "lane_change_duration": 5,
    "disable_tb": DISABLE_TB,
    "disable_ramp_metering": DISABLE_RAMP_METER,
    # For ADDITIONAL_RL_ENV_PARAMS
    "target_velocity": 40,
    "add_rl_if_exit": False
}

# flow rate
flow_rate = 200 * SCALING

# percentage of flow coming out of each lane
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=flow_rate * (1 - AV_FRAC),
    departLane="random",
    departSpeed=0)

# inflow.add(
#     name='followerstopper',
#     veh_type="followerstopper",
#     edge="1",
#     vehs_per_hour=flow_rate * (1 - AV_FRAC),
#     departLane="random",
#     departSpeed=0)

inflow2 = InFlows()
inflow2.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=100 * (1 - AV_FRAC),
    departLane="random",
    departSpeed=0)

inflow3 = InFlows()
inflow3.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=10 * (1 - AV_FRAC),
    departLane="random",
    departSpeed=0)

inflow4 = InFlows()
inflow4.add(
    veh_type="human",
    edge="1",
    vehs_per_hour=50 * (1 - AV_FRAC),
    departLane="random",
    departSpeed=0)

inflow_lst = [inflow, inflow2, inflow3, inflow4]

# inflow.add(
#     veh_type="followerstopper",
#     edge="1",
#     vehs_per_hour=flow_rate * AV_FRAC,
#     departLane="random",
#     departSpeed=10)

traffic_lights = TrafficLightParams()

additional_net_params = {"scaling": SCALING, "speed_limit": args.speed_limit, "length": 985, "width": 100}
net_params = NetParams(
    inflows=inflow,
    additional_params=additional_net_params)

net_params2 = NetParams(
    # inflows=inflow,
    additional_params=additional_net_params)

loop_network1 = CarFollowingNetwork(
    name='car_following',
    vehicles=vehicles,
    net_params=net_params,
    initial_config=InitialConfig(edges_distribution=['1','2']),
    traffic_lights=traffic_lights,
)

loop_network2 = CarFollowingNetwork(
    name='car_following',
    vehicles=vehicles2,
    net_params=net_params2,
    initial_config=InitialConfig(edges_distribution=['1']),
    traffic_lights=traffic_lights,
)


straight_network1 = CarFollowingStraightNetwork(
    name='car_following2',
    vehicles=vehicles,
    net_params=net_params,
    initial_config=InitialConfig(edges_distribution=["1"]),
    traffic_lights=traffic_lights
)

straight_network2 = CarFollowingStraightNetwork(
    name='car_following2',
    vehicles=vehicles2,
    net_params=net_params,
    initial_config=InitialConfig(edges_distribution=["1"]),
    traffic_lights=traffic_lights
)


env_config = {
    "simulator": "traci",

    "sim_params": SumoParams(
        sim_step=0.1,
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),

    "env_params": EnvParams(
        warmup_steps=200,
        sims_per_step=1,
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),
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
                "name": "CarFollowingConnector",
                "params": {}
            },
            "reward_params": {
                "name": "VehRewardConnector",
                "params": {
                    "W1": [1, 1],
                    "W2": [1, 1],
                    "W3": [1, 1],
                    "W4": [args.speed_reward_weight, args.speed_reward_weight] if args.speed_reward else [0, 0],
                    "f_eff_type": args.f_eff_type,
                }
            }
        }
    }
}

env1: CarFollowingEnv = WolfEnv.create_env(
    cls=CarFollowingEnv,
    network=loop_network1,
    env_params=env_config['env_params'],
    sim_params=env_config['sim_params'],
    agents_params=env_config["agents_params"],
    env_state_params=env_config["env_state_params"],
    group_agents_params=env_config["group_agents_params"],
    multi_agent_config_params=env_config["multi_agent_config_params"],
    simulator=env_config["simulator"],
    action_repeat_params=env_config.get("action_repeat_params", None),
)
env2: CarFollowingEnv = WolfEnv.create_env(
    cls=CarFollowingEnv,
    network=loop_network2,
    env_params=env_config['env_params'],
    sim_params=env_config['sim_params'],
    agents_params=env_config["agents_params"],
    env_state_params=env_config["env_state_params"],
    group_agents_params=env_config["group_agents_params"],
    multi_agent_config_params=env_config["multi_agent_config_params"],
    simulator=env_config["simulator"],
    action_repeat_params=env_config.get("action_repeat_params", None),
)


# env1.reset()
# env2.reset()
# for _ in range(5):
#     states, reward, done, infos = env.step({'veh_followerstopper_0': np.array([3])})
    # print('state',len(states.get('veh_followerstopper_0')))
    # print('done',done)

checkpoint_dir = args.save_dir
writer = SummaryWriter(f'runs/ddpg_{unix_timestamp}_{time_string}_eff_type_{args.f_eff_type}')

# Define and build DDPG agent
hidden_size = tuple(args.hidden_size)
agent = DDPG(args.gamma,
                args.tau,
                hidden_size,
                5, ##state space
                np.array([1,]), ##action space
                checkpoint_dir=checkpoint_dir
                )
# Initialize replay memory
memory = ReplayMemory(int(args.replay_size))

# Initialize OU-Noise
# nb_actions = env.action_space.shape[-1]
nb_actions = 1

ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                        sigma=float(args.noise_stddev) * np.ones(nb_actions))


# Define counters and other variables
start_step = 0
# timestep = start_step
if args.load_model:
    # Load agent if necessary
    start_step, _ = agent.load_checkpoint()
timestep = start_step // 10000 + 1
rewards, policy_losses, value_losses, mean_test_rewards, sum_test_rewards = [], [], [], [], []
epoch = 0
t = 0
time_last_checkpoint = time.time()

episode_counter = 0

# Start training
logger.info('Doing {} timesteps'.format(args.timesteps))
logger.info('Start at timestep {0} with t = {1}'.format(timestep, t))
logger.info('Start training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))







for eps in range(args.episodes):
    ou_noise.reset()
    epoch_return = 0
    ## update config

    env = env1
    # if episode_counter > 0 and episode_counter % 20 == 0:
    #     env_index = np.random.randint(2)
    #     env = [env1, env2][env_index]
    #     if env_index == 0: # Only add inflow into the environment that has leading vehicles
    #         env.network.net_params.inflows = random.choice(inflow_lst) # Randomly choose a inflow rate from the network every 20 episodes
    #                                                                    # Currently each episode is 3000 steps (HORIZON)

    state=env.reset()
    Travel_distance=0
    delta_T=0.1
    agent_names = list(state.keys())
    agent_names.sort()
    batch_state = np.array([state[agent_name] for agent_name in agent_names])
    state = torch.Tensor(batch_state).to(device)
    while True:
        
        if state[0][0] > 0:
            Travel_distance=Travel_distance+state[0][0]*delta_T

        # print('headway?',state[0][2])
        # action=[]
        # for i in range(len(state)):
        #     # print('state i ',state[i])
        #     action_cal=agent.calc_action(state[i], ou_noise)
        #     if args.constrain:
        #         action_cal=constrain_action(state[i],action_cal)
                
        #     action.append(list(action_cal))
        action = agent.calc_action(state, ou_noise) # Batch calculation
        accels = action.cpu().numpy()
        # print('action',action)
        
        # print('accel',accel)
        action_dir = dict((agent_name, accel) for agent_name, accel in zip(agent_names, accels))
        next_state, reward, done_dict, _ = env.step(action_dir)

        reward = np.array([reward[agent_name] for agent_name in agent_names])
        # print('reward',reward)
        reward_avg = np.average(reward)
        # print('avg reward',reward_avg)
        done=done_dict.get('__all__')
        # if done:
            # next_state = None
        # else:

        next_agent_names = list(next_state.keys())
        for i, agent_name in enumerate(agent_names):
            if done_dict[agent_name]:
                next_state[agent_name] = state[i].cpu().numpy()
                next_agent_names.remove(agent_name)

        batch_state = np.array([next_state[agent_name] for agent_name in agent_names])

        tmp_next_state = torch.Tensor(batch_state).to(device)

        # print('next state',next_state)
        timestep += 1
        epoch_return += reward_avg

        mask = torch.Tensor([done]).to(device)
        reward = torch.Tensor(reward).to(device)
        # print('reward',reward)
        accels=torch.Tensor(accels).to(device)
        # print('action',accels)
        ## use each agent's obs r action to update ddpg model
        epoch_value_loss = 0
        epoch_policy_loss = 0
        for i in range(len(state)):
            reward_ = torch.Tensor([reward[i]]).to(device)
            accel_ = torch.Tensor([[accels[i]]]).to(device)
            state_= torch.Tensor([list(state[i])]).to(device)
            next_state_= torch.Tensor([list(tmp_next_state[i])]).to(device)

            # print('reward',reward_,0.02*average_speed)
            memory.push(state_, accel_, mask, next_state_, reward_)

            # state[i] = tmp_next_state[i]


            if len(memory) > args.batch_size:
                transitions = memory.sample(args.batch_size)
                # Transpose the batch
                # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

        # for agent_name in done_dict:
        #     if done_dict[agent_name] and agent_name != '__all__':
        #         agent_names.remove(agent_name)
        
        # Reload agent_names based on the 
        # Reconstruct batch_state to prune the agent that is "done"
        agent_names = next_agent_names
        # agent_names.sort(key=lambda s: int(re.findall(r'\d+', s)[0]))
        agent_names.sort()
        batch_state = np.array([next_state[agent_name] for agent_name in agent_names])
        next_state = torch.Tensor(batch_state).to(device)
        state = next_state

        if done:
            episode_counter += 1
            break
    print('travel distance',Travel_distance.cpu().numpy())
    rewards.append(epoch_return)
    value_losses.append(epoch_value_loss)
    policy_losses.append(epoch_policy_loss)
    writer.add_scalar('epoch/return', epoch_return, eps)

    # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
    if eps % args.save_freq==0:
        t += 1
        test_rewards = []
        for _ in range(args.n_test_cycles):
            ## update config
            state=env.reset()
            agent_names = list(state.keys())
            batch_state = np.array([state[agent_name] for agent_name in agent_names])            
            state = torch.Tensor(batch_state).to(device)
            test_reward = 0

            while True:
                if args.render_eval:
                    env.render()

                action = agent.calc_action(state, action_noise=None)
                action=[]
                for i in range(len(state)):
                    # print('state i', state[0][i])
                    # print('len state',len(state))
                    action_cal=agent.calc_action(state[i], action_noise=None)
                    action.append(list(action_cal))

                action=torch.tensor(action)

                accels = action.cpu().numpy()
                if args.constrain:
                    tmp_accels = []
                    for i in range(len(accels)):
                        # print('state i',state[i])
                        tmp_accels.append(constrain_action(state[i], accels[i]))
                    accels = tmp_accels
                
                action_dir = dict((agent_name, accel) for agent_name, accel in zip(agent_names, accels))




                next_state, reward, done_dict, _ = env.step(action_dir)
                
                done=done_dict.get('__all__')
                reward=np.array([reward[agent_name] for agent_name in agent_names])

                test_reward += np.average(reward)

                agent_names = list(next_state.keys())
                for agent_name in done_dict:
                    if done_dict[agent_name] and agent_name != '__all__':
                        agent_names.remove(agent_name)

                batch_state = np.array([next_state[agent_name] for agent_name in agent_names])
                state = torch.Tensor(batch_state).to(device)


                if done:
                    break
            test_rewards.append(test_reward)

        mean_test_rewards.append(np.mean(test_rewards))
        sum_test_rewards.append(np.sum(test_rewards))

        for name, param in agent.actor.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), eps)
        for name, param in agent.critic.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), eps)

        writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], eps)
        writer.add_scalar('test/sum_test_return', sum_test_rewards[-1], eps)
        logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
                    "mean reward: {}, mean test reward {}".format(eps,
                                                                    timestep,
                                                                    rewards[-1],
                                                                    np.mean(rewards[-10:]),
                                                                    np.mean(test_rewards)))

        # Save if the mean of the last three averaged rewards while testing
        # is greater than the specified reward threshold
        # TODO: Option if no reward threshold is given
        # if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
        #     agent.save_checkpoint(timestep, memory)
        #     time_last_checkpoint = time.time()
        #     logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
        # if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
        #     os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        notes_path = os.path.join(checkpoint_dir, 'notes.txt')
        if not os.path.exists(notes_path):
            with open(notes_path, 'w') as f:
                print(f"DDPG, has constraint: {args.constrain}, "+
                f"has speed reward: {args.speed_reward}, "+
                f"speed_reward weight: {args.speed_reward_weight}, "+
                f"f_eff_type: {args.f_eff_type}, seed: {args.seed}, "+
                f"f_follow_ttc: {args.follow_ttc}", file=f)

        agent.save_checkpoint(timestep, memory, last_episode=eps)
        time_last_checkpoint = time.time()
        logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))


if os.path.exists('/home/yifei/its_sow45/tmp.yaml'):
    with open('/home/yifei/its_sow45/tmp.yaml', 'r') as f:
        exp_records = yaml.load(f)
    eff_type_key = f'f_eff_type_{args.f_eff_type}'
    if exp_records[eff_type_key] is None:
        exp_records[eff_type_key] = []
    exp_records[eff_type_key].append({
        'car_follow_5_states_multi_agent_no_constraint_no_speed': {
            'location': 'server',
            'notes': f'DDPG, single agent, state has 5 dim, f_eff type: {args.f_eff_type}',
            'checkpt_dir': checkpoint_dir,
            'highest_train_return': str(np.round(np.max(rewards), decimals=3)),
            'highest_test_return': str(np.round(np.max(mean_test_rewards), decimals=3)),
            'straight perturbation': None,
            'loop': None
        }
    })
    with open('/home/yifei/its_sow45/tmp.yaml', 'w') as f:
        yaml.dump(exp_records, f)

    # epoch += 1

agent.save_checkpoint(timestep, memory, last_episode=eps)
logger.info('Saved model at endtime {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
logger.info('Stopping training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
env.close()