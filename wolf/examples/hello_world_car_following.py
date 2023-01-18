from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND
from wolf.world.environments.wolfenv.car_following_env import CarFollowingEnv, CarFollowingNetwork
from wolf.world.environments.wolfenv.wolf_env import WolfEnv
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams, TrafficLightParams, VehicleParams


from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController, IDMController

import numpy as np

# time horizon of a single rollout
HORIZON = 1000
# number of parallel workers
N_CPUS = 2
# number of rollouts per training iteration
N_ROLLOUTS = N_CPUS * 4

SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.10

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="all_checks",
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=5 * SCALING)


vehicles.add(
    veh_id="followerstopper",
    acceleration_controller=(RLController, {}),
    # lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
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
#     veh_type="followerstopper",
#     edge="1",
#     vehs_per_hour=flow_rate * AV_FRAC,
#     departLane="random",
#     departSpeed=10)

traffic_lights = TrafficLightParams()

additional_net_params = {"scaling": SCALING, "speed_limit": 23, "length": 985, "width": 100}
net_params = NetParams(
    inflows=inflow,
    additional_params=additional_net_params)

network = CarFollowingNetwork(
    name='car_following',
    vehicles=vehicles,
    net_params=net_params,
    initial_config=InitialConfig(),
    traffic_lights=traffic_lights,
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
                "name": "CarFollowingConnector",
                "params": {}
            },
            "reward_params": {
                "name": "VehRewardConnector",
                "params": {}
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

env.reset()
for _ in range(5000):
    env.step({'veh_followerstopper_0': np.array([3])})
