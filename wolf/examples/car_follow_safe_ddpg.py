from wolf.models.safe_layer_ddpg import SafeLayerDDPG
from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController, SimCarFollowingController, GippsController, IDMController
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoParams, EnvParams, NetParams, \
    InitialConfig, DetectorParams, InFlows, TrafficLightParams, SumoLaneChangeParams

from wolf.world.environments.wolfenv.car_following_env import\
     CarFollowingEnv, CarFollowingNetwork, CarFollowingStraightNetwork, CustomizedIDMController, CustomizedGippsController
from wolf.world.environments.wolfenv.wolf_env import WolfEnv
from wolf.examples.utils.noise import OrnsteinUhlenbeckActionNoise
from wolf.examples.utils.replay_memory import ReplayMemory, Transition

import random
import numpy as np
from gym.spaces import Box, Discrete
import argparse
import time
import os
import pickle as pkl
import pandas as pd
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# Generate time string
named_tuple = time.localtime()
time_string = time.strftime("%m-%d_%H:%M", named_tuple)

parser = argparse.ArgumentParser()

parser.add_argument("--render_train", default=False, type=bool,
                    help="Render the training steps (default: False)")
parser.add_argument("--render_eval", default=False, type=bool,
                    help="Render the evaluation steps (default: False)")
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default=f"./tmp/ddpg_{int(time.time())}_{time_string}/",
                    help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed (default: 0)")
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
parser.add_argument("--n_test_cycles", default=10, type=int,
                    help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
parser.add_argument("--constrain-weight", default=1., type=float,
                    help="Coefficient for constraint loss")
# parser.add_argument("--constrain", default=True, action="store_true", help="Enable constrain [Enabled by default]")
# parser.add_argument("--no-constrain", dest="constrain", action="store_false", help="Disable constrain")
parser.add_argument("--speed-reward", default=True, action='store_true')
parser.add_argument("--no-speed-reward", dest="speed_reward", action='store_false')
parser.add_argument("--speed-limit", default=None, type=int)
args = parser.parse_args()

# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

def generate_env(horizon=3000, lead_vh=True, follower_controller='rl', lead_vh_controller='builtin', speed_limit=None, network_type='loop', has_inflow=True):
    # Initial Vehicle Config
    vehicles = VehicleParams()

    if follower_controller == 'rl':
        controller = (RLController, {})
    elif follower_controller == 'gipps':
        controller = (CustomizedGippsController, {"acc": 3})
    elif follower_controller == 'idm':
        controller = (CustomizedIDMController, {"a": 3})
    
    vehicles.add(
        veh_id="followerstopper",
        car_following_params=SumoCarFollowingParams(
            speed_mode="aggressive"
        ),
        acceleration_controller=controller,
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)
    
    if lead_vh:
        if lead_vh_controller == 'builtin':
            vehicles.add(
                veh_id="human",
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
        elif lead_vh_controller == 'dummy':
            raise NotImplementedError

    
    # flow rate
    flow_rate = 200

    # percentage of flow coming out of each lane
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehs_per_hour=flow_rate * (1 - 0.1),
        departLane="random",
        departSpeed=0)
    

    # Traffic Light Config
    traffic_lights = TrafficLightParams()

    # Network Config
    additional_net_params = {"scaling": 1, "speed_limit": speed_limit,
                        "length": 985, "width": 100}

    net_params = NetParams(
        inflows=inflow if has_inflow else None,
        additional_params=additional_net_params)

    if network_type == 'loop':
        network = CarFollowingNetwork(
            name='car_following',
            vehicles=vehicles,
            net_params=net_params,
            initial_config=InitialConfig(edges_distribution=['1']),
            traffic_lights=traffic_lights,
        )
    elif network_type == 'straight':
        network = CarFollowingStraightNetwork(
            name='car_following',
            vehicles=vehicles,
            net_params=net_params,
            initial_config=InitialConfig(edges_distribution=['1']),
            traffic_lights=traffic_lights,
        )
    else:
        raise NotImplementedError

    # Env Config
    # Set up env parameters
    additional_env_params = {
        # For ADDITIONAL_ENV_PARAMS
        "max_accel": 3,
        "max_decel": 3,
        "lane_change_duration": 5,
        # For ADDITIONAL_RL_ENV_PARAMS
        "add_rl_if_exit": False,
        # whether the toll booth should be active
        "disable_tb": True,
        # whether the ramp meter is active
        "disable_ramp_metering": True,
    }


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
            horizon=horizon,
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
                        "W4": [0, 0]
                    }
                }
            }
        }
    }

    # Create environment
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
    env.record_eval = False
    env.trial_id = None

    return env

# Save dirname suffix
dir_name = os.path.basename(os.path.normpath(args.save_dir))
prefix_dir = os.path.dirname(os.path.normpath(args.save_dir))
args.save_dir = os.path.join(prefix_dir, dir_name+f'_lambda_{args.constrain_weight}')

# Create training environment
env = generate_env(has_inflow=True)
eval_env = generate_env(has_inflow=False)
writer = SummaryWriter(f'runs/ddpg_{int(time.time())}_{time_string}')

# ===============================================
EVAL_INTERVAL = 10000
CHECKPT_FREQ = 10000
# Create traning agent
action_space: Box = Box(-3, 3, (1,))
hidden_size = tuple(args.hidden_size)
agent = SafeLayerDDPG(args.gamma, args.tau, hidden_size, 
                      num_inputs=5, action_space=action_space,
                      checkpoint_dir=args.save_dir)
memory = ReplayMemory(int(args.replay_size))

# Initialize OU-Noise
nb_actions = 1
ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                        sigma=float(args.noise_stddev) * np.ones(nb_actions))

agent_name = 'veh_followerstopper_0' # Hardcoded

step_counter = 0
episode_counter = 0
rewards, policy_losses, value_losses = [], [], []

eval_episode_rewards = []
eval_episode_reward = -np.inf
eval_episode_durations = []

device = "cuda" if torch.cuda.is_available() else "cpu"

# Start training
logger.info('Doing {} timesteps'.format(args.timesteps))
logger.info('Start training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

def eval_model(env, agent: SafeLayerDDPG, curr_steps, agent_name=agent_name, device=device):
    state_dir = env.reset() # Reset env
    assert len(state_dir.keys()) == 1, "This env currently supports one agent"
    state = state_dir[agent_name]
    state = torch.Tensor([state]).to(device)
    t = 0
    episode_reward = 0
    while t < 5000:
        t += 1
        action = agent.calc_action(state)
        accel = action.cpu().numpy()[0]

        next_state_dir, reward_dir, done, _ = env.step({agent_name: accel})
        next_state = next_state_dir[agent_name]
        reward = reward_dir[agent_name]

        next_state = torch.Tensor([next_state]).to(device)
        state = next_state # Replace state with next_state
        done = done['__all__']

        episode_reward += reward

        if done:
            print("Total Reward in this evaluation:", episode_reward)
            print("Total Steps in evaluation:", t)
            break
    
    env_record = env.history_record

    # Convert it to pandas DataFrame
    stats = pd.DataFrame(env_record)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f'eval_{curr_steps}.pkl'), 'wb') as f:
        pkl.dump(stats, f)
    
    return episode_reward, t


while step_counter <= args.timesteps:
    ou_noise.reset()
    episode_step_counter= 0
    episode_reward = 0
    travel_distance=0

    state_dir = env.reset()
    assert len(state_dir.keys()) == 1, "This env currently supports one agent"
    state = state_dir[agent_name]
    state = torch.Tensor([state]).to(device)

    while True:
        # Inisde one episode/rollout
        step_counter += 1
        episode_step_counter

        veh_speed = state[0][0]
        if veh_speed > 0:
            travel_distance = travel_distance + veh_speed * 0.1
        if args.render_train:
            env.render()
        
        action = agent.calc_action(state, ou_noise)
        accel = action.cpu().numpy()[0]

        next_state_dir, reward_dir, done, _ = env.step({agent_name: accel})
        
        reward = reward_dir.get(agent_name)
        done = done.get('__all__')
        next_state = next_state_dir.get(agent_name)

        episode_reward += reward
        
        mask = torch.Tensor([done]).to(device)
        reward = torch.Tensor([reward]).to(device)
        next_state = torch.Tensor([next_state]).to(device)

        memory.push(state, action, mask, next_state, reward)

        state = next_state

        if len(memory) > args.batch_size:
            transitions = memory.sample(args.batch_size)

            batch = Transition(*zip(*transitions))

            # Update model
            agent.update_params(batch, args.constrain_weight)


        if step_counter % EVAL_INTERVAL == 0:
            print(f"Evaluating model at step {step_counter}........")

            test_rewards = []
            test_durations = []
            for _ in range(args.n_test_cycles):
                agent.set_eval()
                eval_r, eval_t = eval_model(eval_env, agent, curr_steps=step_counter)
                agent.set_train()

                test_rewards.append(eval_r)
                test_durations.append(eval_t)

            mean_eval_reward = np.mean(test_rewards)
            eval_episode_rewards.append(mean_eval_reward)
            eval_episode_durations.append(np.mean(test_durations))
            # Saving record after each evaluation so we can see the result ealier
            # EVAL_INTERVAL is relatively large so there shouldn't be any IO bottleneck
            os.makedirs(args.save_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, 'eval_durations.pkl'), 'wb') as f:
                pkl.dump(eval_episode_durations, f)

            with open(os.path.join(args.save_dir, 'eval_rewards.pkl'), 'wb') as f:
                pkl.dump(eval_episode_rewards, f)

            # Saving best performance model
            if mean_eval_reward > eval_episode_reward:
                eval_episode_reward = mean_eval_reward
                print("Model with best performance in evaluation updated, saving model.....")
                os.makedirs(args.save_dir, exist_ok=True)
                agent.save_checkpoint(step_counter, memory, eval_flag=True)

            writer.add_scalar('test/mean_test_return', eval_episode_rewards[-1], step_counter)
            writer.add_scalar('test/mean_test_duration', eval_episode_durations[-1],
            step_counter)

        if step_counter % CHECKPT_FREQ == 0:
            # TODO: Augment the file path
            os.makedirs(args.save_dir, exist_ok=True)
            agent.save_checkpoint(step_counter, memory)

        if done:
            episode_counter += 1
            rewards.append(episode_reward)
            writer.add_scalar('epoch/return', episode_reward, episode_counter)
            print('travel distance',travel_distance.cpu().numpy())
            os.makedirs(args.save_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, 'rewards.pkl'), 'wb') as f:
                pkl.dump(rewards, f)
            break