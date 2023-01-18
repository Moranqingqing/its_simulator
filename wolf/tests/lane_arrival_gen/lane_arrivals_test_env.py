import numpy as np

"""
Environment for testing classes derived from LaneArrivals
Mostly based on GenericGridEnv and SimpleGridEnv
(The classes being tested are defined in the lane_arrival_gen module.)
"""

# Import vehicle controllers
from flow.controllers import SimCarFollowingController, GridRouter

# Import parameter-holder objects
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoParams, EnvParams, NetParams, \
    InitialConfig, DetectorParams

# Underlying Traffic Light Control Network
from flow.networks.asymetric_grid_network import TrafficLightAsymetricGridNetwork
from wolf.world.environments.wolfenv.wolf_env import WolfEnv

class LaneArrivalsTestEnv(WolfEnv):
    def __init__(
            self,
            simulator='traci',
            m=3,    # rows
            n=3,    # cols
            horizon=500,
            speed_limit=30,
            sim_params=None,
            env_params=None,
            env_state_params=None,
            net_params=None,
            vehicles_params=None,
            initial_config_params=None,
            detector_params=None,
            tl_params=None,
            agents_params=None,
            action_repeat_params=None,
            group_agents_params=None,
            multi_agent_config_params=None,
            lane_arrival_sched=None,
    ):

        # Less cluttered way of writing down the default configs
        if sim_params is None:
            sim_params = {
                "restart_instance": True,
                "sim_step": 0.1,
                "print_warnings": False,
                "render": False,
            }

        if env_params is None:
            env_params = {
                "horizon": horizon,
                "additional_params": {
                    "target_velocity": speed_limit,
                    "switch_time": 3,
                    "num_observed": 2,
                    "discrete": False,
                    "tl_type": "actuated",
                    "num_local_edges": 4,
                    "num_local_lights": 4,
                },
            }

        if net_params is None:
            net_params = {
                "additional_params": {
                    "speed_limit": speed_limit,
                    "grid_array": {
                        "row_inner_lengths": None,
                        "col_inner_lengths": None,
                        "short_length": 500,    # Gotta be the same as long_length
                        "long_length": 500,     # at the moment...?
                        "cars_left": 0,
                        "cars_right": 0,
                        "cars_top": 0,
                        "cars_bot": 0,
                    },
                    "horizontal_lanes": 1,
                    "vertical_lanes": 1,
                }
            }

        if vehicles_params is None:
            vehicles_params = {
                "min_gap": 3,
                "max_speed": speed_limit,
                "decel": 7.5,
                "speed_mode": "right_of_way",
            }

        if initial_config_params is None:
            initial_config_params = {"spacing": 'custom', "shuffle": True}

        if detector_params is None:
            detector_params = {'positions': [-5, -100], 'frequency': 100}

        if tl_params is None:
            sim_step = sim_params['sim_step']
            phase_min = int(10 / sim_step)
            phase_max = int(60 / sim_step)

            tl_params = {
                "ALL": {
                    "name": "second_based",
                    "params": {
                        "phases": [
                            #{"colors": "rGrG", "min_time": phase_min, "max_time": phase_max},
                            #{"colors": "GrGr", "min_time": phase_min, "max_time": phase_max},
                            {"colors": "GGGG", "min_time": phase_min, "max_time": phase_max},
                        ],
                        "initialization": "random",
                    },
                }
            }

        if agents_params is None:
            agents_params = {
                'name' : 'all_the_same',
                'params' : {
                    'global_reward' : False,
                    'default_policy' : None,

                    'action_params' : {
                        'name' : 'ExtendChangePhaseConnector',
                        'params' : {},
                    },

                    'obs_params' : {
                        'name' : 'TDTSEConnector',
                        'params' :
                        {
                            'obs_params' :
                            {
                                'num_history' : 60,
                                'detector_position' : [5, 100],
                            },
	                    'phase_channel' : True,
                        },
                    },

                    'reward_params' : {
                        'name' : 'QueueRewardConnector',
                        'params' : { 'stop_speed' : 2 }
                    },
                },
            }

        if multi_agent_config_params is None:
            multi_agent_config_params = {
                'name' : 'shared_policy',
                'params' : {},
            }

        # Set down lengths of roads in the grid

        # Deploying eye-sparing methodology...
        short_length = net_params['additional_params']['grid_array']['short_length']
        long_length  = net_params['additional_params']['grid_array']['long_length']

        if net_params['additional_params']['grid_array']['row_inner_lengths'] is None:
            net_params['additional_params']['grid_array']['row_inner_lengths'] = \
	        (m - 1) * [short_length]

        if net_params['additional_params']['grid_array']['col_inner_lengths'] is None:
            net_params['additional_params']['grid_array']['col_inner_lengths'] = \
                (n - 1) * [short_length]

        # Done with defining default configs! Time to initialize
        detector_params_obj = DetectorParams()
        for i in range(m * n):
            detector_params_obj.add_induction_loop_detectors_to_intersection(
                name=f"det_{i}",
                node_id=f"center{i}",
                **detector_params)

        vehicles = VehicleParams()
        # Add a vehicle type
        vehicles.add(
            veh_id='human',
            acceleration_controller=(SimCarFollowingController, {}),
            car_following_params=SumoCarFollowingParams(**vehicles_params),
            routing_controller=(GridRouter, {}),
            num_vehicles=0    # Number of vehicles in the initial configuration
        )

        initial_config = InitialConfig(**initial_config_params)

        network = TrafficLightAsymetricGridNetwork(
            name="lane_arrival_test_env",
            vehicles=vehicles,
            net_params=NetParams(inflows=None, **net_params),
            initial_config=initial_config,
            detector_params=detector_params_obj
        )

        WolfEnv.__init__(
            self=self,
            agents_params=agents_params,
            group_agents_params=group_agents_params,
            multi_agent_config_params=multi_agent_config_params,
            env_params=EnvParams(**env_params),
            sim_params=SumoParams(**sim_params),
            network=network,
            tl_params=tl_params,
            lane_arrival_sched=lane_arrival_sched,
            env_state_params=env_state_params,
            controlled_nodes=None,
            action_repeat_params=action_repeat_params,
            simulator=simulator
        )
