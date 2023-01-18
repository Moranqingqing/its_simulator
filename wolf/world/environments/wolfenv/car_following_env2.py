from flow.networks.bottleneck import BottleneckNetwork
from flow.envs.bottleneck import BottleneckAccelEnv
from flow.networks.bottleneck import ADDITIONAL_NET_PARAMS
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import TrafficLightParams
from flow.controllers.rlcontroller import RLController

import numpy as np
import logging
from copy import deepcopy


# Main purpose to extend this is to have detector_params
# Otherwise it won't be compatible with flow/utils/registry.py
# Line 91 `create_env(*_)`
class CarFollowingNetwork(BottleneckNetwork):
    def __init__(
        self,
        name,
        vehicles,
        net_params,
        initial_config=InitialConfig(),
        traffic_lights=TrafficLightParams(),
        detector_params=None):
       
        """Instantiate the network class."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))
        
        super().__init__(name, vehicles, net_params, initial_config, traffic_lights)


MAX_LANES = 4  # base number of largest number of lanes in the network


class CarFollowingEnv(BottleneckAccelEnv):
    def __init__(
        self,
        env_params,
        sim_params,
        network,
        simulator='traci'
    ):
        super().__init__(env_params, sim_params, network, simulator)
        # Copy of rl_id_list, don't want to bother the original
        # implementation since I don' fully understand the usage
        # of this property
        self.rl_id_list_cp = self.generate_initial_rl_id_list()

    
    def generate_initial_rl_id_list(self):
        """Self-generated the rl_id_list_cp due to the unexpected initialization
        in the TraCIVehicle class
        """
        rl_veh_type_lst = []
        type_parameters = self.k.vehicle.type_parameters
        rl_id_list_cp = []
        
        # Loop through the type_parameters and find the rl vehicle types
        for veh_type in type_parameters:
            if type_parameters[veh_type]['acceleration_controller'][0] == RLController:
                rl_veh_type_lst.append(veh_type)

        print(rl_veh_type_lst)
        # Push the rl_veh ids to self.rl_id_list_cp
        if self.simulator == 'traci':
            for veh_id, veh_config in self.k.vehicle._TraCIVehicle__vehicles.items():
                # Add it to self.rl_id_list_cp, the latter condition
                # is just double check
                print(veh_config)
                if veh_config['type'] in rl_veh_type_lst and\
                     veh_id not in rl_id_list_cp:
                    rl_id_list_cp.append(veh_id)
        
        return rl_id_list_cp

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if rl_actions is not None:
            return super().compute_reward()
        else:
            return 0

    # def reset(self):
    #     super().reset()
        # The original class implementation doens't have this line
        # self.rl_id_list_cp = deepcopy(self.k.vehicle.get_rl_ids())
        # print("Get rl_id_list_cp")
        # print(self.k.vehicle.get_rl_ids())

    def get_state(self):
        """See class definition."""
        headway_scale = 1000

        rl_ids = self.k.vehicle.get_rl_ids()

        # rl vehicle data (absolute position, speed, and lane index)
        rl_obs = np.empty(0)
        id_counter = 0
        for veh_id in rl_ids:
            # check if we have skipped a vehicle, if not, pad
            # If not? You sure it's not the reverse?
            rl_id_num = self.rl_id_list_cp.index(veh_id)
            if rl_id_num != id_counter:
                rl_obs = np.concatenate(
                    (rl_obs, np.zeros(4 * (rl_id_num - id_counter))))
                id_counter = rl_id_num + 1
            else:
                id_counter += 1

            # get the edge and convert it to a number
            edge_num = self.k.vehicle.get_edge(veh_id)
            if edge_num is None or edge_num == '' or edge_num[0] == ':':
                edge_num = -1
            else:
                # I am not sure why they use this '/'
                # Be aware of this
                edge_num = int(edge_num) / 6
            rl_obs = np.concatenate((rl_obs, [
                # absolute position
                self.k.vehicle.get_x_by_id(veh_id) / 1000,
                # speed
                (self.k.vehicle.get_speed(veh_id) / self.max_speed),
                # lane index, edge_number
                (self.k.vehicle.get_lane(veh_id) / MAX_LANES), edge_num
            ]))

        # if all the missing vehicles are at the end, pad
        diff = self.num_rl - int(rl_obs.shape[0] / 4)
        if diff > 0:
            rl_obs = np.concatenate((rl_obs, np.zeros(4 * diff)))

        # relative vehicles data (lane headways, tailways, vel_ahead, and
        # vel_behind)
        relative_obs = np.empty(0)
        id_counter = 0
        for veh_id in rl_ids:
            # check if we have skipped a vehicle, if not, pad
            rl_id_num = self.rl_id_list_cp.index(veh_id)
            if rl_id_num != id_counter:
                pad_mat = np.zeros(
                    4 * MAX_LANES * self.scaling * (rl_id_num - id_counter))
                relative_obs = np.concatenate((relative_obs, pad_mat))
                id_counter = rl_id_num + 1
            else:
                id_counter += 1
            # scaling is fixed across the network, check init of the parent class
            num_lanes = MAX_LANES * self.scaling
            # headway shape: (num_lanes,)
            headway = np.asarray([1000] * num_lanes) / headway_scale
            # tailway shape: (num_lanes,)
            tailway = np.asarray([1000] * num_lanes) / headway_scale
            # vel_in_front shape: (num_lanes,)
            vel_in_front = np.asarray([0] * num_lanes) / self.max_speed
            # vel_behind shape: (num_lanes,)
            vel_behind = np.asarray([0] * num_lanes) / self.max_speed

            lane_leaders = self.k.vehicle.get_lane_leaders(veh_id)
            lane_followers = self.k.vehicle.get_lane_followers(veh_id)
            lane_headways = self.k.vehicle.get_lane_headways(veh_id)
            lane_tailways = self.k.vehicle.get_lane_tailways(veh_id)

            # Notice when we construct headway, we used MAX_LANES assumption
            # Here lane_headways which read from the kernel might have
            # less # of lanes than the maxiumum assumption, thus some
            # lanes which doesn't really exist would be padded
            # (Again, I don't understand why they choose to get headways
            # and tailways for all lanes, isn't vehicle can only drive on 
            # one lane at the same time? Maybe it's for lane changing behavior?)
            headway[0:len(lane_headways)] = (
                np.asarray(lane_headways) / headway_scale)
            tailway[0:len(lane_tailways)] = (
                np.asarray(lane_tailways) / headway_scale)
            for i, lane_leader in enumerate(lane_leaders):
                if lane_leader != '':
                    vel_in_front[i] = (
                        # Get speed of the lane_leader
                        self.k.vehicle.get_speed(lane_leader) / self.max_speed)
            for i, lane_follower in enumerate(lane_followers):
                if lane_follower != '':
                    # Get speed of the lane_follower
                    vel_behind[i] = (self.k.vehicle.get_speed(lane_follower) /
                                     self.max_speed)

            # relative_obs not only store the relative distance
            # it also stores the speed
            relative_obs = np.concatenate((relative_obs, headway, tailway,
                                           vel_in_front, vel_behind))

        # if all the missing vehicles are at the end, pad
        diff = self.num_rl - int(relative_obs.shape[0] / (4 * MAX_LANES))
        if diff > 0:
            relative_obs = np.concatenate((relative_obs,
                                           np.zeros(4 * MAX_LANES * diff)))

        # per edge data (average speed, density)
        edge_obs = []
        for edge in self.k.network.get_edge_list():
            veh_ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(veh_ids) > 0:
                avg_speed = (sum(self.k.vehicle.get_speed(veh_ids)) /
                             len(veh_ids)) / self.max_speed
                density = len(veh_ids) / self.k.network.edge_length(edge)
                edge_obs += [avg_speed, density]
            else:
                edge_obs += [0, 0]

        return np.concatenate((rl_obs, relative_obs, edge_obs))

    # Override the original implementation by replacing the property
    def additional_command(self):
        """Reintroduce any RL vehicle that may have exited in the last step.

        This is used to maintain a constant number of RL vehicle in the system
        at all times, in order to comply with a fixed size observation and
        action space.
        """
        super(BottleneckAccelEnv, self).additional_command()
        # if the number of rl vehicles has decreased introduce it back in
        num_rl = self.k.vehicle.num_rl_vehicles
        if num_rl != len(self.rl_id_list_cp) and self.add_rl_if_exit:
            # find the vehicles that have exited
            diff_list = list(
                set(self.rl_id_list_cp).difference(self.k.vehicle.get_rl_ids()))
            for rl_id in diff_list:
                # distribute rl cars evenly over lanes
                lane_num = self.rl_id_list_cp.index(rl_id) % \
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

    
