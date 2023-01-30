from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController, SimCarFollowingController, GippsController, IDMController, CFMController
from flow.controllers.base_controller import BaseController
from flow.controllers.car_following_models import BCMController
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoParams, EnvParams, NetParams, \
    InitialConfig, DetectorParams, InFlows, TrafficLightParams, SumoLaneChangeParams
from flow.networks.base import Network
from flow.networks.bottleneck import BottleneckNetwork

from wolf.world.environments.wolfenv.wolf_env import WolfEnv
from wolf.config import WOLF_PATH

import pprint
import numpy as np
import logging
from copy import deepcopy
import psutil
import gc
import json
import pandas as pd
import pickle
import os
import time
import random
from wolf.config import WOLF_PATH
from numpy import pi, sin, cos, linspace


ADDITIONAL_NET_PARAMS = {
    # the factor multiplying number of lanes.
    "scaling": 1,
    # edge speed limit
    'speed_limit': 30,
    # length of the network
    'length': 3*985,
    # width of the network
    'width': 100
}


class CarFollowingNetwork(Network):
    def __init__(
        self,
        name,
        vehicles,
        net_params,
        initial_config=InitialConfig(),
        traffic_lights=TrafficLightParams(),
        detector_params=None):

        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self._length = net_params.additional_params.get("length", 985)
        self._width = net_params.additional_params.get("width", 100)

        scaling = net_params.additional_params.get("scaling", 1)
        assert (isinstance(scaling, int)), "Scaling must be an int"
        self._num_lanes = 1 * scaling

        self._speed = net_params.additional_params['speed_limit']
        

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        nodes = [
            {
                "id": "1",
                "x": self._length / 2 + 10,
                "y": self._width / 4
            },  
            {
                "id": "2",
                "x": self._length / 2 + 20,
                "y": 0
            },  
            {
                "id": "3",
                "x": self._length,
                "y": 0
            },  
            {
                "id": "4",
                "x": self._length,
                "y": self._width
            },
            {
                "id": "5",
                "x": 0,
                "y": self._width
            },
            {
                "id": "6",
                "x": 0,
                "y": 0
            },
            {
                "id": "7",
                "x": self._length / 2 - 20,
                "y": 0
            },
            {
                "id": "8",
                "x": self._length / 2 - 10,
                "y": self._width / 4
            },
            # fake nodes used for visualization
            {
                "id": "fake1",
                "x": 0,
                "y": 1
            },
            {
                "id": "fake2",
                "x": 0,
                "y": 2
            }
        ]  # post-merge2
        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        
        r = self._width / 2     # Radius
        shift_width = 20    # Shift width of the entrance and exit
        end_pt_shift_width = 10     # Shift width of the end points of entrance and exit

        edges = [
            {
                "id": "1",
                "from": "1",
                "to": "2",
                "length": (end_pt_shift_width**2 + (self._width / 4)**2)**0.5,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 10
            },
            {
                "id": "2",
                "from": "2",
                "to": "3",
                "length": self._length / 2 - shift_width,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 28
            },
            {
                "id": "3",
                "from": "3",
                "to": "4",
                "shape": [(r*np.cos(t)+self._length, r*np.sin(t)+r) for t in np.linspace(-np.pi/2, np.pi/2, num=80)],
                "length": np.pi * r,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 28
            },
            {
                "id": "4",
                "from": "4",
                "to": "5",
                "length": self._length,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 20
            },
            {
                "id": "5",
                "from": "5",
                "to": "6",
                "shape": [(r*np.cos(t), r*np.sin(t)+r) for t in np.linspace(np.pi/2, 3*np.pi/2, num=80)],
                "length": np.pi * r,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 28
            },
            {
                "id": "6",
                "from": "6",
                "to": "7",
                "length": self._length / 2 - shift_width,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 28
            },
            {
                "id": "7",
                "from": "7",
                "to": "8",
                "length": (end_pt_shift_width**2 + (self._width / 4)**2)**0.5,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 28
            },
            {
                "id": "8",
                "from": "7",
                "to": "2",
                "length": 2 * shift_width,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 28
            }
        ]
        return edges

    def specify_centroids(self, net_params):
        """See parent class."""
        centroids = []
        centroids += [{
            "id": "1",
            "from": None,
            "to": "1",
            "x": -30,
            "y": 0,
        }]
        centroids += [{
            "id": "1",
            "from": "5",
            "to": None,
            "x": 985 + 30,
            "y": 0,
        }]
        return centroids

    def specify_routes(self, net_params):
        """See parent class."""
        num_vehicles = net_params.additional_params.get("rl_vehs", None)
        if num_vehicles:
            rts = {
                "1": ["1", "2", "3", "4", "5", "6", "8"],
                "2": ["2", "3", "4", "5", "6", "8"],
                "8": ["8", "2", "3", "4", "5", "6", "8"]
            }
        else:
            rts = {
                "1": ["1", "2", "3", "4", "5", "6", "7"],
                "2": ["2", "3", "4", "5", "6", "7"],
                "7": ["7"],
                "8": ["8", "2", "3", "4", "5", "6", "8"],
                "followerstopper_0": ["1", "2", "3", "4", "5", "6", "8"],
            }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        return [("1", 0), 
                ("2", (10**2 + (self._width / 4)**2)**0.5+22.6)] 


class CarFollowingRingNetwork(Network):
    def __init__(
        self,
        name,
        vehicles,
        net_params,
        initial_config=InitialConfig(),
        traffic_lights=TrafficLightParams(),
        detector_params=None):

        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        scaling = net_params.additional_params.get("scaling", 1)
        assert (isinstance(scaling, int)), "Scaling must be an int"
        self._num_lanes = 1 * scaling

        self._speed = net_params.additional_params['speed_limit']
        

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        r = length / (2 * pi)

        nodes = [{
            "id": "bottom",
            "x": 0,
            "y": -r
        }, {
            "id": "right",
            "x": r,
            "y": 0
        }, {
            "id": "top",
            "x": 0,
            "y": r
        }, {
            "id": "left",
            "x": -r,
            "y": 0
        }]

        return nodes


    def specify_edges(self, net_params):
        """See parent class."""
        length = 2400
        resolution = 40
        r = length / (2 * pi)
        edgelen = length / 4


        edges = [
            {
                "id": "bottom",
                "from": "bottom",
                "to": "right",
                "shape": [(r * cos(t), r * sin(t))  for t in linspace(-pi / 2, 0, resolution)],
                "length": edgelen,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 23
            },
            {
                "id": "right",
                "from": "right",
                "to": "top",
                "shape": [(r * cos(t), r * sin(t))  for t in linspace(0, pi / 2, resolution)],
                "length": edgelen,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 23
            },
            {
                "id": "top",
                "from": "top",
                "to": "left",
                "shape": [(r * cos(t), r * sin(t)) for t in linspace(pi / 2, pi, resolution)],
                "length": edgelen,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 23
            },
            {
                "id": "left",
                "from": "left",
                "to": "bottom",
                "shape": [(r * cos(t), r * sin(t))  for t in linspace(pi, 3 * pi / 2, resolution)],
                "length": edgelen,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 15
            },
        ]
        return edges

    def specify_centroids(self, net_params):
        """See parent class."""
        centroids = []
        centroids += [{
            "id": "1",
            "from": None,
            "to": "1",
            "x": -30,
            "y": 0,
        }]
        centroids += [{
            "id": "1",
            "from": "5",
            "to": None,
            "x": 985 + 30,
            "y": 0,
        }]
        return centroids

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "top": ["top", "left", "bottom", "right"],
            "left": ["left", "bottom", "right", "top"],
            "bottom": ["bottom", "right", "top", "left"],
            "right": ["right", "top", "left", "bottom"]
        }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        ring_length = 1200
        junction_length = 0  # length of inter-edge junctions

        edgestarts = [("bottom", 0),
                      ("bottom", 10),
                    #   ("top", 0.5 * ring_length + 2 * junction_length),
                    #   ("left", 0.75 * ring_length + 3 * junction_length)]
        ]
        return edgestarts




class CarFollowingStraightNetwork(Network):
    def __init__(
        self,
        name,
        vehicles,
        net_params,
        initial_config=InitialConfig(),
        traffic_lights=TrafficLightParams(),
        detector_params=None):

        scaling = net_params.additional_params.get("scaling", 1)
        assert (isinstance(scaling, int)), "Scaling must be an int"
        self._num_lanes = 1 * scaling
        self._speed = net_params.additional_params.get("speed_limit", None)

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):

        nodes = [
            {
                "id": "1",
                "x": 0,
                "y": 0
            },
            {
                "id": "2",
                "x": 300,
                "y": 0
            },
            {
                "id": "3",
                "x": 700,
                "y": 0
            },
            {
                "id": "4",
                "x": 1100,
                "y": 0
            },
            {
                "id": "5",
                "x": 1600,
                "y": 0
            },
            {
                "id": "6",
                "x": 2400,
                "y": 0
            },
            {
                "id": "7",
                "x": 3000,
                "y": 0
            },
            {
                "id": "8",
                "x": 3500,
                "y": 0
            },
        ]
        return nodes

    def specify_edges(self, net_params):
        edges = [
            {
                "id": "1",
                "from": "1",
                "to": "2",
                "length": 300,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 10
            },
            {
                "id": "2",
                "from": "2",
                "to": "3",
                "length": 400,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 15
            },
            {
                "id": "3",
                "from": "3",
                "to": "4",
                "length": 400,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 17.5
            },
            {
                "id": "4",
                "from": "4",
                "to": "5",
                "length": 500,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 20
            },
            {
                "id": "5",
                "from": "5",
                "to": "6",
                "length": 800,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 23
            },
            {
                "id": "6",
                "from": "6",
                "to": "7",
                "length": 600,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 20
            },
            {
                "id": "7",
                "from": "7",
                "to": "8",
                "length": 500,
                "spreadType": "center",
                "numLanes": self._num_lanes,
                "speed": self._speed if self._speed else 18
            },
        ]
        return edges

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "1": ["1", "2", "3", "4", "5", "6", "7"],
            "2": ["2", "3", "4", "5", "6", "7"],
            "7": ["7"]
        }
        return rts

    def specify_edge_starts(self):
        """See parent class."""
        return [("1", 0), ("2", 100.1)]
    
        
# time horizon of a single rollout
HORIZON = 36000
# number of parallel workers
N_CPUS = 2
# number of rollouts per training iteration
# N_ROLLOUTS = N_CPUS * 4

SCALING = 1
NUM_LANES = 1 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.001

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 5,
    # whether the toll booth should be active
    "disable_tb": True,
    # whether the ramp meter is active
    "disable_ramp_metering": True,
}


MAX_LANES = 1  # base number of largest number of lanes in the network
# Edge 1 is the entrance, edge 7 is the exit, edge 8 connects edge 2 and edge 6
EDGE_LIST = ["1", "2", "3", "4", "5", "6", "7", "8"]
BOTTLE_NECK_LEN = 280  # Length of bottleneck



class CarFollowingEnv(WolfEnv):
    def __init__(
        self,
        agents_params,
        group_agents_params,
        multi_agent_config_params,
        env_params,
        sim_params,
        network,
        tl_params={},
        env_state_params=None,
        action_repeat_params=None,
        simulator='traci'
    ):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        
        self.agents_params = agents_params

        # Initialize the class with TrafficEnv
        super().__init__(
            agents_params=agents_params,
            group_agents_params=group_agents_params, 
            multi_agent_config_params=multi_agent_config_params,
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            tl_params=tl_params)

        # ===================================================================
        # Initialize from bottleneck.Bottleneck
        env_add_params = self.env_params.additional_params
        # tells how scaled the number of lanes are
        self.scaling = network.net_params.additional_params.get("scaling", 1)
        self.edge_dict = dict()
        # =====================================================================
        # Accel part from bottleneck.BottleneckAccel
        self.add_rl_if_exit = env_params.get_additional_param("add_rl_if_exit")
        self.num_rl = deepcopy(self.initial_vehicles.num_rl_vehicles)
        self.rl_id_list = deepcopy(self.generate_initial_rl_id_list())
        self.max_speed = self.k.network.max_speed()
        self.record_train = False
        self.record_eval =False
        self.reward_folder='/home/tianyushi/code/uoft-research/its_sow45/record/record-idm' ##manually set record folder
        # TODO: Change this
        # Temp solution to save the acceleration performacne
        self.history_record = {}
        self.global_metric = {}
        self.initial_distance = {}
        self.initial_gap = None

    def reset(self):
        self.logger.debug("Resetting Env")

        def extra_work():
            if psutil.virtual_memory().percent >= 60.0:
                gc.collect()

            self.logger.debug("Performing extra work in flow reset function")
            if self._agents is None:
                return

            if 'veh_fake_0' in self.get_agents():
                self.deregister_agent('veh_fake_0')

            for id, agent in self.get_agents().items():
                agent.reset()
            
            # Reset history record
            self.history_record = {}
        
        obs = super().reset(perform_extra_work=extra_work)
        return obs
    
    def get_state(self):
        if self._agents is None:
            return {}
        obs = super().get_state()
        return obs

    def additional_command(self):
        super().additional_command()

        # build a dict containing the list of vehicles and their position for
        # each edge and for each lane within the edge
        empty_edge = [[] for _ in range(MAX_LANES * self.scaling)]

        self.edge_dict = {k: deepcopy(empty_edge) for k in EDGE_LIST}
        for veh_id in self.k.vehicle.get_ids():
            try:
                edge = self.k.vehicle.get_edge(veh_id)
                if edge not in self.edge_dict:
                    self.edge_dict[edge] = deepcopy(empty_edge)
                lane = self.k.vehicle.get_lane(veh_id)  # integer
                pos = self.k.vehicle.get_position(veh_id)
                self.edge_dict[edge][lane].append((veh_id, pos))
            except Exception:
                pass
        
        # Remove the agent that is not on the road anymore
        agents_to_del = []
        if self._agents is not None:
            for agent_id in self._agents:
                # agent id != veh_id saved in connector
                agent_veh_id = agent_id.lstrip('veh_')
                # TODO: Need to check the agent type (veh or tl, etc.)
                if agent_veh_id not in self.k.vehicle.get_rl_ids():
                    agents_to_del.append(agent_id)
            for agent_id in agents_to_del:
                self.deregister_agent(agent_id)

        # ===============================================================
        # From BottleneckAccel
        # if the number of rl vehicles has decreased introduce it back in
        num_rl = self.k.vehicle.num_rl_vehicles
        if num_rl != len(self.rl_id_list) and self.add_rl_if_exit:
            # find the vehicles that have exited
            diff_list = list(
                set(self.rl_id_list).difference(self.k.vehicle.get_rl_ids()))
            for rl_id in diff_list:
                # distribute rl cars evenly over lanes
                lane_num = self.rl_id_list.index(rl_id) % \
                           MAX_LANES * self.scaling
                # reintroduce it at the start of the network
                try:
                    self.k.vehicle.add(
                        veh_id=rl_id,
                        edge='1',
                        type_id=str('rl'),
                        lane=str(lane_num),
                        pos="0",
                        speed="max")
                except Exception:
                    pass
        # ===============================================================

        # Add the agent that is on the road but not in the env._agents
        agent_ids = set()
        if self._agents is not None:
            agent_ids = set(agent_id.lstrip('veh_') for agent_id in self._agents.keys())
        new_rl_veh_ids = set(self.k.vehicle.get_rl_ids()) - agent_ids
        if len(new_rl_veh_ids) > 0:
            from wolf.utils.configuration.registry import R
            factory = R.agent_factory(self.agents_params["name"])
            agents = factory.create_agents_by_ids(self, new_rl_veh_ids, **self.agents_params["params"])
            # Filter out the non-necessary agents
            # TODO: Add some method in the agent_factory (i.e. create_agent_by_ids)
            self.register_agents(agents)


    # NOTE: Crash checking was turned off in the original parent method
    # For details, check `MultiEnv._step_helper()` and `crash = 0`
    def _step_helper(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : dict of array_like
            agent's observation of the current environment
        reward : dict of floats
            amount of reward associated with the previous state/action pair
        done : dict of bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    accel_contr = self.k.vehicle.get_acc_controller(veh_id)
                    action = accel_contr.get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicle in the
            # network, including rl and sumo-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(veh_id)
                    routing_actions.append(route_contr.choose_route(self))
            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()
            # crash = 0
            # stop collecting new simulation steps if there is a collision
            if crash:
                break
        
        states = self.get_state()
        done = {key: key.lstrip('veh_') in self.k.vehicle.get_arrived_rl_ids()
                for key in states.keys()}
        if crash or (self.time_counter >= self.env_params.sims_per_step *
                     (self.env_params.warmup_steps + self.env_params.horizon)):
            done['__all__'] = True
        else:
            done['__all__'] = False
        infos = {key: {} for key in states.keys()}

        # compute the reward
        if self.env_params.clip_actions:
            clipped_actions = self.clip_actions(rl_actions)
            reward = self.compute_reward(clipped_actions, fail=crash)
        else:
            reward = self.compute_reward(rl_actions, fail=crash)

        
        for rl_id in self.k.vehicle.get_arrived_rl_ids():
            done['veh_'+rl_id] = True
            reward['veh_'+rl_id] = 0
            # states['veh_'+rl_id] = np.array([-1001., 1001., -1001., 1001., -1001., -1001., 0.]) #TODO Nicolas did this instead of this "np.zeros(self.observation_space.shape[0])"
        
        # if set(self.get_veh_rl_ids()) == set(self.k.vehicle.get_arrived_rl_ids()):
        if len(self.get_veh_rl_ids()) > 0 and set(self.get_veh_rl_ids()) == set(self.k.vehicle.get_arrived_rl_ids()):
            done['__all__'] = True

        # TODO: Remove this
        self.record_metrics(crash, done['__all__'] if '__all__' in done else False)

        return states, reward, done, infos

    def record_metrics(self, crash, done):
        """ Record metrics and save the metrics detail """
        if self.record_train:
            if crash or (self.time_counter >= self.env_params.sims_per_step *
                        (self.env_params.warmup_steps + self.env_params.horizon)):
                for agent_name, tmp_agent in self._agents.items():
                    stats = {'f_ttc': tmp_agent._reward_connector.f_ttc_arr,
                             'f_eff': tmp_agent._reward_connector.f_eff_arr,
                             'f_jerk': tmp_agent._reward_connector.f_jerk_arr,
                             'ttc': tmp_agent._reward_connector.ttc_arr,
                             'eff': tmp_agent._reward_connector.eff_arr,
                             'jerk': tmp_agent._reward_connector.jerk_arr,
                             'total_reward': tmp_agent._reward_connector.total_reward_arr,
                             'action': tmp_agent.actions}
                    
                    pkl_file_path = os.path.join(self.reward_folder, f'reward_records_{int(time.time())}.pkl')
                    if os.path.exists(pkl_file_path):
                        # My attempt to fix the concurrency issue 
                        # Add 1 sec if another file with same name is just saved.
                        pkl_file_path = os.path.join(self.reward_folder,
                        f'reward_records_{int(time.time())+1}.pkl')
                    with open(pkl_file_path, 'wb') as f:
                        pickle.dump(stats, f)

        # Use this to reduce indentation level
        if not self.record_eval:
            return

        # For eval case, we want to save all vehicles records, so for many vehicles we have to
        # calculate again
        if len(self.k.vehicle.get_ids()) > 0 and not crash:
            # Checking crash should avoid the case that speed < 0, we just ignore the metrics
            # from that step, be aware that I am not 100% sure tho

            speeds = self.k.vehicle.get_speed(self.k.vehicle.get_ids())
            if speeds:
                if 'mean_speed' in self.global_metric:
                    self.global_metric['mean_speed'].append(np.mean(speeds))
                else:
                    self.global_metric['mean_speed'] = [np.mean(speeds)]
            
            outflow = self.k.vehicle.get_outflow_rate(50)
            inflow = self.k.vehicle.get_inflow_rate(50)

            if 'outflow' in self.global_metric:
                self.global_metric['outflow'].append(outflow)
            else:
                self.global_metric['outflow'] = [outflow]
            
            if 'inflow' in self.global_metric:
                self.global_metric['inflow'].append(inflow)
            else:
                self.global_metric['inflow'] = [inflow]

            delay = []
            rel_distance = []
            for veh_id, veh_speed in zip(self.k.vehicle.get_ids(), speeds):
                edge_num = self.k.vehicle.get_edge(veh_id)
                speed_limit = self.k.network.speed_limit(edge_num)
                delay.append((speed_limit - veh_speed) / speed_limit)
                
                distance = self.k.vehicle.get_distance(veh_id)
                if self.initial_gap is None:
                    headway = self.k.vehicle.get_headway(veh_id)
                    if 0 < headway < 100:
                        self.initial_gap = headway
                if veh_id not in self.initial_distance:
                    self.initial_distance[veh_id] = distance
                # FIXME: speed_limit - 5 is a tmp solution since the dummy controller of lv for perturbation test
                # only change speed in +/-5 patterns
                expect_distance = self.initial_distance[veh_id] + (speed_limit - 5) * (self.time_counter-1) * 0.1
                d_offset = distance - expect_distance
                if self.initial_gap is None:
                    initial_gap = 0
                else:
                    initial_gap = self.initial_gap
                rel_distance.append(d_offset + initial_gap*len(rel_distance)) #TODO: Change 25, this is hardcoded now for init veh_space
            
            if 'delay' in self.global_metric:
                self.global_metric['delay'].append(np.mean(delay))
            else:
                self.global_metric['delay'] = [np.mean(delay)]
            
            if 'time_space' in self.global_metric:
                self.global_metric['time_space'].append(rel_distance)
            else:
                self.global_metric['time_space'] = [rel_distance]
            

            for veh_id in self.k.vehicle.get_ids():
                speed = self.k.vehicle.get_speed(veh_id)

                accel = None
                if ((self.time_counter <= self.env_params.warmup_steps+1 and \
                    veh_id in self.k.vehicle.get_rl_ids()) or \
                    (veh_id not in self.k.vehicle.get_controlled_ids() and \
                    veh_id not in self.k.vehicle.get_rl_ids())) and veh_id in self.history_record:
                    # The case that rl vehicle is in warmup stage and no way to get acc connectors.
                    # Or the vehicle is controlled by some builtin controller that don't return acc output
                    # (Named `human drivers` in flow)
                    # We also need to guarantee there is at least 1 record of this vehicle to take advantage
                    # of the following formula

                    # We delayed calculate the acceleration
                    prev_speed = self.history_record[veh_id]['speed'][-1]

                    # Use (v_t - v_{t-1}) / delta_t to calculate the acc
                    accel = (speed - prev_speed) / self.sim_params.sim_step
                elif veh_id in self.k.vehicle.get_controlled_ids():
                    # If we can get the the accel directly from the acc_controller
                    # (i.e. check gipps, notice rl_controller won't have this so we need to 
                    # manually handle it)
                    acc_controller = self.k.vehicle.get_acc_controller(veh_id)
                    accel = acc_controller.get_action(self)
                
                position = self.k.vehicle.get_distance(veh_id)
                curr_edge = self.kernel.get_edge(veh_id)
                speed_limit = self.kernel.get_speed_limit_by_edge(curr_edge)

                lead_veh = self.k.vehicle.get_leader(veh_id)
                lead_veh_speed = np.nan
                gap = np.nan
                time_headway = np.nan
                ttc = np.nan
                target_hw = np.nan
                if lead_veh is not None:
                    # Some corner case (i.e., around joints), use `get_lane_leaders` instead
                    gap = self.k.vehicle.get_headway(veh_id)
                    lead_veh_speed = self.k.vehicle.get_speed(lead_veh)
                    target_hw=0.207 * np.log(lead_veh_speed) +0.226
                    # Calculate time_headway
                    if speed > 0:
                        time_headway = gap / speed
                    else:
                        # Case that time_headway
                        time_headway = np.inf
                    
                    # Calculate ttc (time-to-collision)
                    if speed > lead_veh_speed:
                        # Case that vehicle is faster than lead vehicle's speed
                        ttc = gap / (speed - lead_veh_speed)
                    else:
                        # Case that vehicle is slower than lead vehicle's speed
                        ttc = np.inf
                
                follow_veh = self.k.vehicle.get_follower(veh_id)
                follow_veh_speed = np.nan
                follow_distance_headway = np.nan
                follow_time_headway = np.nan
                follow_ttc = np.nan
                if follow_veh is not None:
                    follow_distance_headway = self.k.vehicle.get_headway(follow_veh)
                    follow_veh_speed = self.k.vehicle.get_speed(follow_veh)

                    if follow_veh_speed > 0:
                        follow_time_headway = follow_distance_headway / follow_veh_speed
                    else:
                        follow_time_headway = np.inf
                    
                    if follow_veh_speed > speed:
                        follow_ttc = gap / (follow_veh_speed - speed)
                    else:
                        follow_ttc = np.inf

                # Save record to the corresponding object
                if veh_id in self.history_record:

                    self.history_record[veh_id]['speed'].append(speed)
                    self.history_record[veh_id]['ttc'].append(ttc)
                    self.history_record[veh_id]['time_headway'].append(time_headway)
                    self.history_record[veh_id]['gap'].append(gap)
                    self.history_record[veh_id]['position'].append(position)
                    self.history_record[veh_id]['lv_speed'].append(lead_veh_speed)
                    self.history_record[veh_id]['speed_limit'].append(speed_limit)
                    self.history_record[veh_id]['follow_time_headway'].append(follow_time_headway)
                    self.history_record[veh_id]['follow_distance_headway'].append(follow_distance_headway)
                    self.history_record[veh_id]['follow_ttc'].append(follow_ttc)
                    self.history_record[veh_id]['follow_veh_speed'].append(follow_veh_speed)
                    self.history_record[veh_id]['target_hw'].append(target_hw)

                    if accel is not None:
                        self.history_record[veh_id]['accel'].append(accel)
                else:
                    self.history_record[veh_id] = {'speed': [np.nan]*(self.time_counter-1) + [speed],
                                                   'accel': [np.nan]*(self.time_counter-1) if accel is None else [np.nan]*(self.time_counter-1) + [accel],
                                                   'ttc': [np.nan]*(self.time_counter-1) + [ttc],
                                                   'time_headway': [np.nan]*(self.time_counter-1) + [time_headway],
                                                   'target_hw': [np.nan]*(self.time_counter-1) + [time_headway],
                                                   'gap': [np.nan]*(self.time_counter-1) + [gap],
                                                   'position': [np.nan]*(self.time_counter-1) + [position],
                                                   'lv_speed': [np.nan]*(self.time_counter-1) + [lead_veh_speed],
                                                   'speed_limit': [np.nan]*(self.time_counter-1) + [speed_limit],
                                                   'follow_time_headway': [np.nan]*(self.time_counter-1) + [follow_time_headway],
                                                   'follow_distance_headway': [np.nan]*(self.time_counter-1) + [follow_distance_headway],
                                                   'follow_ttc': [np.nan]*(self.time_counter-1) + [follow_ttc],
                                                   'follow_veh_speed': [np.nan]*(self.time_counter-1) + [follow_veh_speed],
                                                   }

        if done or crash or self.time_counter >= self.env_params.sims_per_step * \
            (self.env_params.warmup_steps + self.env_params.horizon):
            # Episodes end, dump all records

            # Replace rl agents accelerations
            if self._agents is not None:
                for agent in self._agents:
                    veh_id = agent.lstrip('veh_')
                    # Concat the accel list from connectors
                    self.history_record[veh_id]['accel'] += [action[0] for action in self._agents[agent].actions]

            # Calculate Jerk
            for veh_id, record in self.history_record.items():
                accel_record = record['accel']
                accel_record = [a if a is not None else 0 for a in accel_record]
                accels_diff = np.ediff1d(accel_record)
                jerk = accels_diff / self.sim_params.sim_step
                self.history_record[veh_id]['jerk'] = list(jerk)

            # Padding the unknown value with np.nan
            for veh_id, records in self.history_record.items():
                for record_name, record in records.items():
                    if len(record) < self.time_counter:
                        pad_len = self.time_counter - len(record)
                        records[record_name] = record + [np.nan] * pad_len

            # Dump all files to the specific folder
            try:
                record_path = os.environ['SOW_RECORD_PATH'] #TODO set path
                # record_path = '/home/tianyushi/code/uoft-research/its_sow45/record'

            except KeyError:
                print("[Warning]: Assign Variable SOW_RECORD_PATH to enabel record")
                return

            if self.trial_id:
                eval_record_dir_path = os.path.join(record_path, f'eval_record_{self.trial_id}')
            else:
                eval_record_dir_path = os.path.join(record_path, f'eval_record_{int(time.time())}')

            if not os.path.exists(eval_record_dir_path):
                os.makedirs(eval_record_dir_path)

            # Convert it to pandas DataFrame
            stats = pd.DataFrame(self.history_record)

            pkl_file_path = os.path.join(eval_record_dir_path, f'eval_record_{int(time.time())}.pkl')
            with open(pkl_file_path, 'wb') as f:
                pickle.dump(stats, f)

    def get_bottleneck_density(self, lanes=None):
        """Return the density of specified lanes.

        If no lanes are specified, this function calculates the
        density of all vehicles on all lanes of the bottleneck edges.
        """
        bottleneck_ids = self.k.vehicle.get_ids_by_edge(['3', '4'])
        if lanes:
            veh_ids = [
                veh_id for veh_id in bottleneck_ids
                if str(self.k.vehicle.get_edge(veh_id)) + "_" +
                str(self.k.vehicle.get_lane(veh_id)) in lanes
            ]
        else:
            veh_ids = self.k.vehicle.get_ids_by_edge(['3', '4'])
        return len(veh_ids) / BOTTLE_NECK_LEN

    def generate_initial_rl_id_list(self):
        """Self-generated the rl_id_list_cp due to the unexpected initialization
        in the TraCIVehicle class
        """
        rl_veh_type_lst = []
        type_parameters = self.kernel._flow_kernel.vehicle.type_parameters
        rl_id_list_cp = []
        
        # Loop through the type_parameters and find the rl vehicle types
        for veh_type in type_parameters:
            if type_parameters[veh_type]['acceleration_controller'][0] == RLController:
                rl_veh_type_lst.append(veh_type)

        print(rl_veh_type_lst)
        # Push the rl_veh ids to self.rl_id_list_cp
        if self.simulator == 'traci':
            for veh_id, veh_config in self.kernel._flow_kernel.vehicle._TraCIVehicle__vehicles.items():
                # Add it to self.rl_id_list_cp, the latter condition
                # is just double check
                print(veh_config)
                if veh_config['type'] in rl_veh_type_lst and\
                     veh_id not in rl_id_list_cp:
                    rl_id_list_cp.append(veh_id)
        
        return rl_id_list_cp

    def get_veh_rl_ids(self):
        return self.rl_id_list

class ClosedRoadNetCarFollowing(CarFollowingEnv):
    """ A car following env on the closed road network """
    def __init__(
        self,
        agents_params,
        group_agents_params, # Could be None
        multi_agent_config_params,
        sim_params=None,
        horizon=HORIZON,
        tl_params={},
        env_state_params=None,
        action_repeat_params=None,
        simulator='traci',
        record_flag=False,
        reward_folder=None
    ):  
        self.reward_folder = reward_folder
        self.record_eval = True
        self.record_train = record_flag # From yaml config file ##TODO: Modify this to info metrics 
        # Initial Vehicle Config
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="human",
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="all_checks",
                max_speed=50
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode= 0b010101011001,
            ),
            num_vehicles=0) #5 * SCALING)

        vehicles.add(
            veh_id="followerstopper",
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="right_of_way",
                accel=3,
                decel=3,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode= 0b010101011001),
            num_vehicles=0) #1 * SCALING) ##bug?? why over 5 will collide!

        # Inflow Vehicles config
        # flow rate
        flow_rate = 1000 * SCALING
        # percentage of flow coming out of each lane
        inflow = InFlows()
        # inflow.add(
        #     veh_type="human",
        #     edge=458575912,
        #     vehs_per_hour=flow_rate * (1 - AV_FRAC),
        #     departLane="random",
        #     departSpeed=0)

        inflow.add(
                    veh_type="human",
                    edge='4255.36', ### circle ramp
                    vehs_per_hour=900,
                    departLane="random",
                    departSpeed=0)
        inflow.add(
                    veh_type="followerstopper",
                    edge='4255.36',
                    vehs_per_hour=1200,  ## side ramp
                    departLane="random",
                    departSpeed=0)

        inflow.add(
                    veh_type="human",
                    edge='4304', ## main
                    vehs_per_hour=12000,
                    departLane="random",
                    departSpeed=0)

        inflow.add(
                    veh_type="followerstopper",
                    edge='4304', ## main
                    vehs_per_hour=12000,
                    departLane="random",
                    departSpeed=0)


        # inflow.add(
        #             veh_type="human",
        #             edge='merge',
        #             vehs_per_hour=flow_rate * (1 - AV_FRAC),
        #             depart_lane=0,
        #             depart_speed=0)

        # inflow.add(
        #             veh_type="human",
        #             edge='freeway',
        #             vehs_per_hour=flow_rate * (1 - AV_FRAC) / 2,
        #             depart_lane="random",
        #             depart_speed=0)


        # Traffic Light Config
        traffic_lights = TrafficLightParams()
        # Network Config
        additional_net_params = {"scaling": SCALING, "speed_limit": 23,
                                 "length": 985, "width": 100}
        net_params = NetParams(
            inflows=inflow,
            template={'net': os.path.join(WOLF_PATH, 'sumo_net', 'sumo_qew_new', 'churchill_sumo_transfer.net.xml'), ## change the folder and file name here
                      'rou': os.path.join(WOLF_PATH, 'sumo_net', 'sumo_qew_new', 'churchill_sumo_transfer.rou.xml'),
                      'vtype': os.path.join(WOLF_PATH, 'sumo_net', 'sumo_qew_new', 'churchill_sumo_transfer.rou.xml'),
                     },
            additional_params=additional_net_params)
#        network = CarFollowingNetwork(
#            name='car_following',
#            vehicles=vehicles,
#            net_params=net_params,
#            initial_config=InitialConfig(),
#            traffic_lights=traffic_lights,
#        )
        network = Network(
            name='car_following',
            vehicles=vehicles,
            net_params=net_params,
            initial_config=InitialConfig(),
            traffic_lights=traffic_lights,
        )
        # Env Config
        # Set up env parameters
        additional_env_params = {
            # For ADDITIONAL_ENV_PARAMS
            "max_accel": 3,
            "max_decel": 3,
            "lane_change_duration": 5,
            "disable_tb": DISABLE_TB,
            "disable_ramp_metering": DISABLE_RAMP_METER,
            # For ADDITIONAL_RL_ENV_PARAMS
            "add_rl_if_exit": False
        }
        # Generate the env_params
        env_params = dict(
            warmup_steps=200,
            sims_per_step=1,
            horizon=horizon,
            additional_params=additional_env_params
        )
        if sim_params is None:
            sim_params = dict(
                sim_step=0.1,
                render=False,
                print_warnings=False,
                restart_instance=True
            )
        super().__init__(
            agents_params=agents_params,
            group_agents_params=group_agents_params,
            multi_agent_config_params=multi_agent_config_params,
            env_params=EnvParams(**env_params),
            sim_params=SumoParams(**sim_params),
            network=network,
            tl_params=tl_params,
            env_state_params=env_state_params,
            action_repeat_params=None,
            simulator=simulator
        )






class ClosedRoadNetCarFollowingEval(CarFollowingEnv):
    """ A car following env on the closed road network """
    
    def __init__(
        self,
        agents_params,
        group_agents_params, # Could be None
        multi_agent_config_params,
        sim_params=None,
        horizon=HORIZON,
        tl_params={},
        env_state_params=None,
        action_repeat_params=None,
        simulator='traci',
        trial_id=None
    ):

        self.record_eval = True
        self.record_train = False # Should always be False in eval
        self.trial_id = trial_id

        # Initial Vehicle Config
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="human",
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="all_checks",
                accel=3,
                decel=3,
                max_speed=23
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=5 * SCALING)

        vehicles.add(
            veh_id="followerstopper",
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive"
            ),
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=1 * SCALING)

        # vehicles.add(
        #     veh_id="gipps",
        #     car_following_params=SumoCarFollowingParams(
        #         speed_mode="aggressive",
        #         accel=3,
        #         decel=3,
        #         max_speed=23
        #     ),
        #     acceleration_controller=(GippsController, {"acc": 3, "v0": 23}),
        #     routing_controller=(ContinuousRouter, {}),
        #     lane_change_params=SumoLaneChangeParams(
        #         lane_change_mode=0,
        #     ),
        #     num_vehicles=1 * SCALING)

        # Inflow Vehicles config
        # flow rate
        flow_rate = 20 * SCALING

        # percentage of flow coming out of each lane
        inflow = InFlows()
        inflow.add(
            veh_type="human",
            edge="1",
            vehs_per_hour=flow_rate * (1 - AV_FRAC),
            departLane="random",
            departSpeed=0)
        

        # Traffic Light Config
        traffic_lights = TrafficLightParams()

        # Network Config
        additional_net_params = {"scaling": SCALING, "speed_limit": 23,
                                 "length": 985, "width": 100, "rl_vehs": 2}

        net_params = NetParams(
            # inflows=inflow,
            additional_params=additional_net_params)

        network = CarFollowingNetwork(
            name='car_following',
            vehicles=vehicles,
            net_params=net_params,
            initial_config=InitialConfig(),
            traffic_lights=traffic_lights,
        )

        # Env Config
        # Set up env parameters
        additional_env_params = {
            # For ADDITIONAL_ENV_PARAMS
            "max_accel": 3,
            "max_decel": 3,
            "lane_change_duration": 5,
            "disable_tb": DISABLE_TB,
            "disable_ramp_metering": DISABLE_RAMP_METER,
            # For ADDITIONAL_RL_ENV_PARAMS
            "add_rl_if_exit": False
        }

        # Generate the env_params
        env_params = dict(
            warmup_steps=200,
            sims_per_step=1,
            horizon=horizon,
            additional_params=additional_env_params
        )

        if sim_params is None:
            sim_params = dict(
                sim_step=0.1,
                render=False,
                print_warnings=False,
                restart_instance=True
            )

        super().__init__(
            agents_params=agents_params,
            group_agents_params=group_agents_params,
            multi_agent_config_params=multi_agent_config_params,
            env_params=EnvParams(**env_params),
            sim_params=SumoParams(**sim_params),
            network=network,
            tl_params=tl_params,
            env_state_params=env_state_params,
            action_repeat_params=None,
            simulator=simulator
        )


class ClosedRoadNetCarFollowingEval1(CarFollowingEnv):
    """ A car following env on the closed road network """
    
    def __init__(
        self,
        agents_params,
        group_agents_params, # Could be None
        multi_agent_config_params,
        sim_params=None,
        horizon=1400,
        tl_params={},
        env_state_params=None,
        action_repeat_params=None,
        simulator='traci',
        trial_id=None
    ):

        # Set random seed for leading vehicle behavior
        # This might have effects on other stochastic behavior (i.e., inflow, etc.)
        random.seed(0)
        np.random.seed(0)

        self.record_eval = True
        self.record_train = False # Should always be False in eval
        self.trial_id = trial_id

        # Initial Vehicle Config
        vehicles = VehicleParams()

        vehicles.add(
            veh_id="human_follow",
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="all_checks",
                accel=3,
                decel=3,
                max_speed=60
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=3 * SCALING)

        # vehicles.add(
        #     veh_id="followerstopper",
        #     car_following_params=SumoCarFollowingParams(
        #         speed_mode="aggressive"
        #     ),
        #     acceleration_controller=(RLController, {}),
        #     routing_controller=(ContinuousRouter, {}),
        #     num_vehicles=1 * SCALING)

        # vehicles.add(
        #     veh_id="gipps",
        #     car_following_params=SumoCarFollowingParams(
        #         speed_mode="aggressive",
        #         accel=3,
        #         decel=3,
        #     ),
        #     acceleration_controller=(CustomizedGippsController, {"acc": 3, "v0": 23}),
        #     routing_controller=(ContinuousRouter, {}),
        #     lane_change_params=SumoLaneChangeParams(
        #         lane_change_mode=0,
        #     ),
        #     num_vehicles=1 * SCALING)

        # vehicles.add(
        #     veh_id="idm",
        #     car_following_params=SumoCarFollowingParams(
        #         speed_mode="aggressive",
        #         accel=3,
        #         decel=3,
        #     ),
        #     acceleration_controller=(CustomizedIDMController, {"a": 3, "v0": 23, "T": 0.6}),
        #     routing_controller=(ContinuousRouter, {}),
        #     lane_change_params=SumoLaneChangeParams(
        #         lane_change_mode=0,
        #     ),
        #     num_vehicles=1 * SCALING)        


        vehicles.add(
            veh_id="human",
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="all_checks",
                accel=3,
                decel=3,
                max_speed=60
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=1 * SCALING)


        # vehicles.add(
        #     veh_id="gipps",
        #     car_following_params=SumoCarFollowingParams(
        #         speed_mode="aggressive",
        #         accel=3,
        #         decel=3,
        #         max_speed=23
        #     ),
        #     acceleration_controller=(GippsController, {"acc": 3, "v0": 23}),
        #     routing_controller=(ContinuousRouter, {}),
        #     lane_change_params=SumoLaneChangeParams(
        #         lane_change_mode=0,
        #     ),
        #     num_vehicles=1 * SCALING)
        

        # Traffic Light Config
        traffic_lights = TrafficLightParams()

        # Network Config
        additional_net_params = {"scaling": SCALING}

        net_params = NetParams(
            # inflows=inflow,
            additional_params=additional_net_params)

        network = CarFollowingStraightNetwork(
            name='car_following_straight',
            vehicles=vehicles,
            net_params=net_params,
            initial_config=InitialConfig(edges_distribution=["1"]), # We only want the vehicle be place in the first edge
            traffic_lights=traffic_lights,
        )

        # Env Config
        # Set up env parameters
        additional_env_params = {
            # For ADDITIONAL_ENV_PARAMS
            "max_accel": 3,
            "max_decel": 3,
            "lane_change_duration": 5,
            "disable_tb": DISABLE_TB,
            "disable_ramp_metering": DISABLE_RAMP_METER,
            # For ADDITIONAL_RL_ENV_PARAMS
            "add_rl_if_exit": False
        }

        # Generate the env_params
        env_params = dict(
            warmup_steps=50,
            sims_per_step=1,
            horizon=horizon,
            additional_params=additional_env_params
        )

        if sim_params is None:
            sim_params = dict(
                sim_step=0.1,
                render=False,
                print_warnings=False,
                restart_instance=True
            )

        super().__init__(
            agents_params=agents_params,
            group_agents_params=group_agents_params,
            multi_agent_config_params=multi_agent_config_params,
            env_params=EnvParams(**env_params),
            sim_params=SumoParams(**sim_params),
            network=network,
            tl_params=tl_params,
            env_state_params=env_state_params,
            action_repeat_params=None,
            simulator=simulator
        )

class CustomizedGippsController(GippsController):
    """ Customized class for dynamic speed limit on different edges """
    def get_accel(self, env: WolfEnv):

        # Get current edge        
        curr_edge = env.kernel.get_edge(self.veh_id)
        # Get corresponding speed limit on the edge
        speed_limit = env.kernel.get_speed_limit_by_edge(curr_edge)
        # Reset the desired speed of the controller
        self.v_desired = speed_limit

        return super().get_accel(env)

class CustomizedIDMController(IDMController):
    """ Customized class for dynamic speed limit on different edges """
    def get_accel(self, env: WolfEnv):
        # Get current edge        
        curr_edge = env.kernel.get_edge(self.veh_id)
        # Get corresponding speed limit on the edge
        speed_limit = env.kernel.get_speed_limit_by_edge(curr_edge)
        # Reset the desired speed of the controller
        self.v0 = speed_limit

        return super().get_accel(env)


class CustomizedBCMController(BCMController):
    """ BCM Controller Implementation from this paper
    """
    def get_accel(self, env: WolfEnv):
        # Get current edge        
        curr_edge = env.kernel.get_edge(self.veh_id)
        # Get corresponding speed limit on the edge
        speed_limit = env.kernel.get_speed_limit_by_edge(curr_edge)
        # Reset the desired speed of the controller
        self.v_des = speed_limit
        return super().get_accel(env)

class CustomizedDummyController(BCMController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_d=1,
                 k_v=1,
                 k_c=1,
                 d_des=1,
                 v_des=8,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        super().__init__(veh_id, car_following_params, k_d=k_d, k_v=k_v, k_c=k_c, d_des=d_des, v_des=v_des, time_delay=time_delay, noise=noise, fail_safe=fail_safe)
        self.decel_flag = True

    def get_accel(self, env: WolfEnv):
        v = env.kernel.get_vehicle_speed(self.veh_id)
        if v <= self.v_des - 5:
            self.decel_flag = False
        elif v >= self.v_des + 5:
            self.decel_flag = True
        if self.decel_flag:
            return -3
        else:
            return 3

class BasicCFMController(CFMController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_d=1,
                 k_v=1,
                 k_c=1,
                 d_des=1,
                 v_des=8,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        super().__init__(veh_id, car_following_params, k_d, k_v, k_c, d_des, v_des, time_delay, noise, fail_safe)


    def get_accel(self, env):
        # Get current edge        
        curr_edge = env.kernel.get_edge(self.veh_id)
        # Get corresponding speed limit on the edge
        speed_limit = env.kernel.get_speed_limit_by_edge(curr_edge)
        # Reset the desired speed of the controller
        self.v_des = speed_limit

        lead_id = env.k.vehicle.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.max_accel

        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        d_l = env.k.vehicle.get_headway(self.veh_id)

        return self.k_d*(d_l - self.d_des) + self.k_v*(lead_vel - this_vel) + \
            self.k_c*(self.v_des - this_vel)

class RLVehRouter(ContinuousRouter):
    """Temp Fix for multi agent training error.
    Extension to the Continuous Router.
    Usage
    -----
    See base class for usage example.
    """

    def choose_route(self, env):
        """See parent class."""
        
        edge = env.k.vehicle.get_edge(self.veh_id)
        if type(env.network) == CarFollowingNetwork:
            if edge == "1":
                return ["1", "2", "3", "4", "5", "6", "8"]
            elif edge == "2":
                return ["2", "3", "4", "5", "6", "8"]

        return super().choose_route(env)