from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController, SimCarFollowingController, GippsController, IDMController, CFMController
from flow.controllers.base_controller import BaseController
from flow.controllers.car_following_models import BCMController
from flow.core.params import (
    VehicleParams,
    SumoCarFollowingParams, SumoLaneChangeParams, SumoParams,
    AimsunCarFollowingParams, AimsunLaneChangeParams, AimsunParams,
    EnvParams, NetParams, InitialConfig, InFlows, TrafficLightParams, AimsunTrafficLightParams
)

# DetectorParams
from flow.networks.base import Network
from flow.networks.bottleneck import BottleneckNetwork

import flow.config as config

from wolf.world.environments.wolfenv.wolf_env import WolfEnv
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


ADDITIONAL_NET_PARAMS = {
    # the factor multiplying number of lanes.
    "scaling": 1,
    # edge speed limit
    'speed_limit': 30,
    # length of the network
    'length': 985,
    # width of the network
    'width': 100
}

# time horizon of a single rollout
HORIZON = 5000
# number of parallel workers
N_CPUS = 2
# number of rollouts per training iteration
# N_ROLLOUTS = N_CPUS * 4

SCALING = 1
NUM_LANES = 1 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.1 #0.01

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
        simulator='aimsun'
    ):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.agents_params = agents_params

        # Initialize the class with WolfEnv
        super().__init__(
            agents_params=agents_params,
            group_agents_params=group_agents_params, 
            multi_agent_config_params=multi_agent_config_params,
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            tl_params=tl_params,
            simulator=simulator)

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

            """
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
            """

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            #if self.sim_params.render:
            #    self.k.vehicle.update_vehicle_colors()

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
                     (self.sim_params.warmup_time + self.sim_params.horizon)):
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
                        (self.sim_params.warmup_steps + self.sim_params.horizon)):
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
                delay.append(abs(speed_limit - veh_speed) / speed_limit)

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
                if ((self.time_counter <= self.sim_params.warmup_steps+1 and \
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
                if lead_veh is not None:
                    # Some corner case (i.e., around joints), use `get_lane_leaders` instead
                    gap = self.k.vehicle.get_headway(veh_id)
                    lead_veh_speed = self.k.vehicle.get_speed(lead_veh)

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

                    if accel is not None:
                        self.history_record[veh_id]['accel'].append(accel)
                else:
                    self.history_record[veh_id] = {'speed': [np.nan]*(self.time_counter-1) + [speed],
                                                   'accel': [np.nan]*(self.time_counter-1) if accel is None else [np.nan]*(self.time_counter-1) + [accel],
                                                   'ttc': [np.nan]*(self.time_counter-1) + [ttc],
                                                   'time_headway': [np.nan]*(self.time_counter-1) + [time_headway],
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
            (self.sim_params.warmup_steps + self.sim_params.horizon):
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
                print("[Warning]: Assign Variable SOW_RECORD_PATH to enable record")
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


class QEWCarFollowing(CarFollowingEnv):
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
        simulator='aimsun',
        record_flag=False,
        reward_folder=None
    ):
        self.reward_folder = reward_folder
        self.record_eval = False
        self.record_train = record_flag # From yaml config file ##TODO: Modify this to info metrics

        # Simulator-dependent Config
        if sim_params is None:
            sim_params = dict(
                sim_step=0.1,
                warmup_time=0,
                horizon=horizon,
                render=False,
                print_warnings=False,
                restart_instance=False
            )

        sim_params = AimsunParams(**sim_params)

        lane_change_params = AimsunLaneChangeParams()
        human_car_following_params=AimsunCarFollowingParams()
        followerstopper_car_following_params = AimsunCarFollowingParams()

        # Initial Vehicle Config
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="human",
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=human_car_following_params,
            lane_change_params=lane_change_params,
            num_vehicles=0, #50 * SCALING,
            simulator=simulator)

        vehicles.add(
            veh_id="followerstopper",
            car_following_params=followerstopper_car_following_params,
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=0, #10 * SCALING,
            simulator=simulator)

        # Inflow Vehicles config
        # flow rate
        flow_rate = 1000 * SCALING

        # percentage of flow coming out of each lane
        inflows = InFlows()

        """
        inflows.add(
            veh_type="human",
            edge=4304,
            vehs_per_hour=flow_rate * (1 - AV_FRAC))

        inflows.add(
            veh_type="human",
            edge=4254,
            vehs_per_hour=flow_rate * (1 - AV_FRAC))
        """

        inflows.add(
            veh_type='followerstopper',
            edge=4304,
            vehs_per_hour=1800 #flow_rate * AV_FRAC
        )


        # Traffic Light Config
        traffic_lights = AimsunTrafficLightParams()

        # Network Config
        additional_net_params = {"scaling": SCALING, "speed_limit": 23,
                                 "length": 985, "width": 100}
        net_params = NetParams(
            inflows=inflows,
            template=os.path.join(config.PROJECT_PATH, 'flow', 'utils', 'aimsun', 'templates', 'qew_no_demand.ang'),
            additional_params=additional_net_params)

        network = Network(
            name='car_following_QEW',
            vehicles=vehicles,
            net_params=net_params,
            initial_config=InitialConfig(),
            traffic_lights=traffic_lights
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
            sims_per_step=1,
            additional_params=additional_env_params
        )

        super().__init__(
            agents_params=agents_params,
            group_agents_params=group_agents_params,
            multi_agent_config_params=multi_agent_config_params,
            env_params=EnvParams(**env_params),
            sim_params=sim_params,
            network=network,
            tl_params=tl_params,
            env_state_params=env_state_params,
            action_repeat_params=None,
            simulator=simulator
        )



# ==== Controllers ====

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
        return super().get_accel(env)

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
