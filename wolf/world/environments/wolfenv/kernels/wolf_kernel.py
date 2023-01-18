from wolf.world.environments.wolfenv.kernels.tl_wolf_kernel import TrafficLightsWolfKernel
import logging
from collections import defaultdict
from wolf.utils.enums import DetectorActivation
from flow.controllers.rlcontroller import RLController


class WolfKernel:

    def __init__(self, flow_kernel, flow_network, sim_params, traffic_lights_params, controlled_nodes=None):
        """
        """
        self.logger = logging.getLogger(__name__)
        self._flow_kernel = flow_kernel
        self._flow_network = flow_network
        self._sim_params = sim_params
        self._traffic_lighted_nodes = None
        self._controlled_nodes = None
        self._set_controlled_nodes(controlled_nodes)
        self.traffic_lights = TrafficLightsWolfKernel(self, traffic_lights_params)


    def get_vehicle_in_detection_range(self, edge_group, leftover_length):
        """
        Get the vehicles covered by the detection range in a certain edge_group.

        Args:
            edge_group (list): edges corresponding to one approach.
            leftover_length (float): left length of the detection range.
                Please check _kernel.get_upstream_edge_groups_with_length

        Returns:
            list: vehicle IDs.
        """
        # previous [:-1] edges are all covered by the detection range
        veh_ids = self.get_vehicle_ids_by_edge(edge_group[:-1])
        # the last edge may not be entirely covered by  the detection range
        pending_veh_ids = self.get_vehicle_ids_by_edge(edge_group[-1])

        if leftover_length >= 0:
            veh_ids += pending_veh_ids
        else:
            for veh in pending_veh_ids:
                pending_veh_pos = self.get_vehicle_position(veh)
                if pending_veh_pos > -leftover_length:
                    veh_ids.append(veh)

        return veh_ids

    def get_ids_by_lane(self, lane_id):
        return self._flow_kernel.vehicle.get_ids_by_lane(lane_id)

    def _set_controlled_nodes(self, node_ids: list):
        """
        Record the IDs of the controlled traffic lights. If do not identify, set to all traffic lights in the network.

        Args:
            node_ids (list): list of traffic lights' IDs
        """
        self._traffic_lighted_nodes = self.get_nodes_ids()
        if node_ids == None:
            self._controlled_nodes = self._traffic_lighted_nodes
        else:
            assert set(node_ids) <= set(self._traffic_lighted_nodes), \
                "Some of the controlled nodes do not exist, please check again."
            self._controlled_nodes = node_ids

    def get_controlled_nodes_ids(self):
        return self._controlled_nodes

    def get_nodes_ids(self):
        return list(self._flow_kernel.traffic_light.get_ids())

    def get_vehicle_speed(self, veh_ids):
        return self._flow_kernel.vehicle.get_speed(veh_ids)

    def get_vehicle_position(self, veh_ids):
        return self._flow_kernel.vehicle.get_position(veh_ids)

    def get_vehicle_ids_by_edge(self, incoming_edges):
        return self._flow_kernel.vehicle.get_ids_by_edge(incoming_edges)

    def get_vehicle_ids_by_lanes(self, lane_ids):
        return self._flow_kernel.vehicle.get_ids_by_lane(lane_ids)

    def set_traffic_light_phase(self, node_id, current_phase):
        self._flow_kernel.traffic_light.set_state(
            node_id=node_id, state=current_phase.colors)

    def get_traffic_light_phase(self, node_id):
        return self._flow_kernel.traffic_light.get_state(node_id=node_id)

    def set_traffic_light_logic(self, node_id, cycle_logic, program_id='0'):
        self._flow_kernel.traffic_light.set_program_logic(node_id, cycle_logic, program_id)

    def get_traffic_light_logic(self, node_id, program_idx=0):
        return self._flow_kernel.traffic_light.get_program_logic(node_id, program_idx)

    def get_traffic_light_incoming_lanes(self, node_id):
        return self._flow_kernel.traffic_light.get_incoming_lanes(node_id)

    def get_traffic_light_incoming_edges(self, node_id):
        return self._flow_kernel.traffic_light.get_incoming_edges(node_id)

    def get_traffic_light_incoming_edge_lanes(self, node_id):
        return self._flow_kernel.traffic_light.get_incoming_edge_lanes(node_id)

    def get_traffic_light_lane_movements(self, node_id):
        return self._flow_kernel.network.get_traffic_light_lane_movements(node_id)

    def get_traffic_light_movement_lane(self, node_id):
        return self._flow_kernel.network.get_traffic_light_movement_lane(node_id)

    def get_edges_lane_numbers(self, edge_ids):
        return self._flow_kernel.traffic_light.get_edges_lane_numbers(edge_ids)

    def get_last_step_vehicle_count_over_detector(self, detector_id):
        return self._flow_kernel.detector.get_last_step_vehicle_count(detector_id)

    def get_vehicle_data_over_detector(self, detector_id):
        return self._flow_kernel.detector.kernel_api.inductionloop.getVehicleData(detector_id)

    def get_detectors_on_lane(self, lane):
        return self._flow_kernel.detector.get_detectors_on_lane(lane)

    def get_detectors_on_edge(self, edge):
        return self._flow_kernel.detector.get_detectors_on_edge(edge)

    def get_ordered_detector_groups(self, node_id, group_count=2, right_turn=False):
        """
        Creates ordered detector groups.

        Args:
            node_id (str): Intersection ID.
            group_count (int, optional): Number of detector groups to look for. Defaults to 2.
            right_turn (bool, optional): Whether to look for right turn detectors. Defaults to False.

        Returns:
            dict: Ordered detector groups.
            First index corresponds to lanes. Second nested index corresponds to different detector groups.

        >>> ordered_detector_groups
        >>> {
                '1175109_0': {
                    0: {'id': 'center1_east_dn_1', 'position': 4.31, 'lane_id': '1175109_0'},
                    1: {'id': 'center1_east_up1_1', 'position': 6.76, 'lane_id': '87950198_0'},
                },
                '1175109_1': {
                    0: {'id': 'center1_east_dn_2', 'position': 4.34, 'lane_id': '1175109_1'},
                    1: {'id': 'center1_east_up1_2', 'position': 6.04, 'lane_id': '87950198_1'},
                },
                '1175109_2': {
                    0: {'id': 'center1_east_dn_3', 'position': 4.47, 'lane_id': '1175109_2'},
                    1: None,
                },
                ...

                # if the upstream group has more detectors than the downstream one
                # add pseudo-lanes for them
                '1307311_extra-2': {
                    0: None,
                    1: {'id': 'center5_south_up1_2', 'position': 9.52, 'lane_id': '1307303_2'},
                }

                # if the incoming edge has right-turn specific lanes
                # add pseudo-lanes for them
                '1307311_rt-0': {
                    0: {'id': 'center5_south_rt_0', 'position': 2.49, 'lane_id': '49280819_4928_0', 'edge_type': 'right_turn'},
                    1: None,
                }
            }
        """
        ordered_detector_groups = defaultdict(dict)
        incoming_edges = self.get_traffic_light_incoming_edges(node_id)
        incoming_edge_lanes = self.get_traffic_light_incoming_edge_lanes(
            node_id)

        for incoming_edge in incoming_edges:
            detector_groups = self.get_edge_detector_groups(
                incoming_edge, group_count, right_turn)

            # num of detectors for each edge equal to the max. #detectors among all the groups
            # **excluding right-turn edges**
            group_detector_counts = [len(list(filter(lambda x: x.get("edge_type", None) != "right_turn", group))) for group in detector_groups.values()]
            detector_count = max(group_detector_counts)
            # and it should not be smaller than the num of controlled lanes
            num_incoming_lane = len(incoming_edge_lanes[incoming_edge])
            assert detector_count >= num_incoming_lane, \
                f"Number of detectors per group should be larger than or equal to the number of controlled lanes. node: {node_id}, edge: {incoming_edge}."

            for det_idx in range(detector_count):
                for grp_idx in range(group_count):
                    # add None to fill the missed detectors
                    detector = detector_groups[grp_idx][det_idx] if \
                        (det_idx < len(detector_groups[grp_idx]) and grp_idx != 0) or \
                        (det_idx < num_incoming_lane and grp_idx == 0) else None
                    # if upstream set has more detectors than downsteam set, add pseudo-lanes for them
                    lane_id = incoming_edge_lanes[incoming_edge][det_idx] if \
                        det_idx < num_incoming_lane else incoming_edge + "_extra-" + str(det_idx)
                    ordered_detector_groups[lane_id][grp_idx] = detector

            # finally, add pseudo-lanes for right-turn specific edge
            for i in range(len(detector_groups[0]) - num_incoming_lane):
                lane_id = incoming_edge + "_rt-" + str(i)
                detectors = [detector_groups[0][num_incoming_lane + i]] + [None] * (group_count - 1)
                ordered_detector_groups[lane_id] = dict(zip(list(range(group_count)), detectors))

        return ordered_detector_groups

    def get_edge_detector_groups(self, starting_edge, group_count, right_turn=False):
        """
        Creates detector groups going backwards from given edge.
        Generally the given edge is an intersection incoming edge.

        Args:
            starting_edge (str): Edge ID.
            group_count (int): Number of detector groups to look for.
            right_turn (bool, optional): Whether to look for right turn detectors. Defaults to False.

        Returns:
            dict: Detector groups.

        >>> detector_groups
        >>> {
                0: [
                    {'id': '905693_east_dn_1', 'position': 4.31, 'lane_id': '1175109_0'},
                    {'id': '905693_east_dn_2', 'position': 4.34, 'lane_id': '1175109_1'},
                    {'id': '905693_east_dn_3', 'position': 4.47, 'lane_id': '1175109_2'}
                ],
                1: [
                    {'id': '905693_east_up1_1', 'position': 6.76, 'lane_id': '87950198_0'},
                    {'id': '905693_east_up1_2', 'position': 6.04, 'lane_id': '87950198_1'}
                ]
            }
        """
        detector_groups = defaultdict(list)
        current_group_index = 0
        edge = starting_edge

        while current_group_index < group_count:
            detectors = self.get_detectors_on_edge(edge)
            if detectors:
                current_group_index = self._update_detector_groups(
                    detectors, current_group_index, detector_groups)
                current_group_index += 1

            prev_edges = self.get_adjacent_edges(edge, 'prev')
            # TODO： this is a **UGLY HARD FIX** for wujiang network.
            # Find a way to fix this !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if edge == "center1_EAST_3.28":
                prev_edges = set([":gneJ191_2"])
            # HARD FIX end.
            edge = prev_edges.pop() if prev_edges else None

        if right_turn:
            right_turn_edge = self.get_right_turn_edge(starting_edge)
            detectors = self.get_detectors_on_edge(right_turn_edge)
            detectors = [dict(x, **{"edge_type": "right_turn"}) for x in detectors]
            detector_groups[0].extend(detectors)

        return detector_groups

    def get_intersection_edge_detector_groups(self, node_id, incoming_edges, num_detector_group=2):
        """
        Get the downstream and upstream detector groups for each edge.

        Args:
            node_id (str): Intersection ID.
            incoming_edges (list): List of IDs of incoming edges of an intersection.
            num_detector_group (int): Number of detector groups this intersecton has.

        Returns:
            dict: Detector groups (2 groups) corresponding to each edge.

            >>> edge_detector_groups
            >>> {
                    '1307309': [
                        ['center5_east_dn_1', 'center5_east_dn_2', 'center5_east_dn_3', 'center5_east_rt_0'],
                        ['center5_east_up1_1', 'center5_east_up1_2', 'center5_east_up1_3']],
                    '1307311': [
                        ['center5_south_dn_1', 'center5_south_dn_2', 'center5_south_rt_0'],
                        ['center5_south_up1_0', 'center5_south_up1_1', 'center5_south_up1_2']],
                    ...
                }
        """
        lane_detectors = self.get_ordered_detector_groups(node_id, num_detector_group, right_turn=True)
        edge_detector_groups = defaultdict(lambda: [[], []])

        for lane, detectors in lane_detectors.items():
            edge = lane.rsplit('_', 1)[0]
            assert edge in incoming_edges, f"Invalid edge id: {edge}"
            if detectors[0] is not None:
                edge_detector_groups[edge][0].append(detectors[0]["id"])
            if detectors[num_detector_group - 1] is not None:
                edge_detector_groups[edge][1].append(detectors[num_detector_group - 1]["id"])

        return edge_detector_groups

    def _update_detector_groups(self, detectors, current_group_index, detector_groups):
        """
        Handles the case of GridEnv where detectors list may contain detectors from different detector groups.
        This method breaks them down into different detector groups.

        Args:
            detectors (list): List of detectors.
            current_group_index (int): Current detector group index.
            detector_groups (dict): Detector groups.
        """
        visited_lanes = set()

        for det in detectors:
            if det['lane_id'] in visited_lanes:
                current_group_index += 1
                detector_groups[current_group_index].append(det)
            else:
                detector_groups[current_group_index].append(det)
                visited_lanes.add(det['lane_id'])

        return current_group_index

    def get_adjacent_edges(self, edge, adj_type):
        """
        Returns the set of adjacent edges. These could be either previous edges or next edges based on type provided.
        Edges can have multiple adjacent connections, therefore set is returned.

        Args:
            edge (str): Edge ID.
            adj_type (str): Can be 'next' or 'prev'.

        Returns:
            set: Set of adjacent edge ID strings.

        >>> adj_conns
        >>> {0: [(':center1_0', 0)],
             1: [(':center1_0', 1)],
             2: [(':center1_6', 0), (':center1_11', 0), (':center1_19', 0)]}

            keys (0, 1, 2) are the lanes of the current edge
            values are the adjacent lanes corresponding to the lane of the key

        >>> adj_edges
        >>> {':center1_0', ':center1_6', ':center1_11', ':center1_19'}
        """
        adj_conns = self._flow_kernel.network._connections[adj_type].get(edge, {})
        adj_edges = set()
        for conn in adj_conns.values():
            edges = list(map(lambda c: c[0], conn))
            adj_edges.update(edges)
        return adj_edges

    def get_right_turn_edge(self, incoming_edge, levels=2):
        """
        Go (levels) number of edges backwards.
        Go forwards the same of number of times, to find the right turn edge.

        Args:
            incoming_edge (str): Incoming edge ID.
            levels (int, optional): Number of edges to look backwards. Defaults to 2.

        Returns:
            str: Right turn edge ID.
        """
        level_edge_mapping = {}
        level_edge_mapping[levels] = {incoming_edge}

        for l in reversed(range(levels)):
            level_edge_mapping[l] = set()
            for e in level_edge_mapping[l + 1]:
                prev_edges = self.get_adjacent_edges(e, 'prev')
                level_edge_mapping[l].update(prev_edges)

        for l in range(levels):
            for e in level_edge_mapping[l]:
                next_edges = self.get_adjacent_edges(e, 'next')
                level_edge_mapping[l + 1].update(next_edges)

        level_edge_mapping[levels].remove(incoming_edge)
        return level_edge_mapping[levels].pop()

    def get_lanes(self):
        pass

    def get_phases(self):
        pass

    def get_green_phases(self):
        pass

    def get_controlled_intersections(self):
        pass

    def _get_upstream_edge_group_with_length(self, edge_id: str, total_length: float):
        """
        For a given edge trace back upstream edges till reached the detection range (total_length)
        or an traffic-lighted intersection.

        Args:
            edge_id (str): edge ID
            total_length (float): detection range.

        Returns:
            edge_group (list[str]): upstream edge group for the given edge.
            leftover_length (float): depends on the situation:
                leftover_length >= 0: detection range is too far, the total length of the edge_group is
                    shorter than the detection range. In this case, leftover_length represents the ignored
                    detection range.
                leftover_length < 0: detection range is fully used. And the last edge in the edge_group
                    does not be entirely covered by the detection range. In this case, leftover_length
                    represents the negative value of exceeded length of that last edge.
        """
        edge_group = [edge_id]
        total_length -= self._flow_kernel.network._edges[edge_id]["length"]
        while total_length > 0:
            incoming_edges = list(self.get_adjacent_edges(edge_id, "prev"))
            if len(incoming_edges) == 0:
                # no upstream edges, reach the origin edge
                return edge_group, total_length

            if len(incoming_edges) == 1:
                # only one incoming edge, keep going backward
                edge_id = incoming_edges[0]
                edge_group.append(edge_id)
                total_length -= self._flow_kernel.network._edges[edge_id]["length"]

            else:
                # upstream edges in a junction, only trace the edge with 's' movement
                junction_id = incoming_edges[0][1:].rsplit('_', 1)[0]
                if self._flow_kernel.network.get_node_type(junction_id) in ['traffic_light', 'traffic_light_unregulated', 'traffic_light_right_on_red']:
                    # tracing back will stop once meet a traffic light
                    return edge_group, total_length
                else:
                    # only trace the edge with 's' movement
                    straight_incoming_edges = self._flow_kernel.network.get_straight_upstream_internal_edges(edge_id)
                    if len(straight_incoming_edges) > 1:
                        print(f"wolf.wolfenv.kernel._get_upstream_edge_group_with_length: edge {edge_id} has multiple straight upstream edges: {straight_incoming_edges}. Tracing back terminated.")
                        return edge_group, total_length

                    edge_id = straight_incoming_edges[0]
                    edge_group.append(edge_id)
                    total_length -= self._flow_kernel.network._edges[edge_id]["length"]

        return edge_group, total_length

    def get_upstream_edge_groups_with_length(self, edge_ids: list, length: float):
        """
        Creates upstream edge groups based on detection range for a set of edges.

        Args:
            edge_ids (list): edge IDs.
            length (float): detection range.

        Returns:
            dict{str: dict}: dictionary of edge groups.
        """
        upstream_edge_groups = dict.fromkeys(edge_ids)
        for edge in edge_ids:
            edge_group, leftover_length = self._get_upstream_edge_group_with_length(edge, length)
            upstream_edge_groups[edge] = {"edge_group": edge_group, "leftover_length": leftover_length}

        return upstream_edge_groups

    def _get_edge_upstream_lane_groups_with_length(self, edge_id: str, length: float):
        upstream_edge_group, leftover_length = self._get_upstream_edge_group_with_length(edge_id, length)
        single_edge_lane_groups = {}

        # match lanes for adjacent edges
        edges_lane_numbers = self.get_edges_lane_numbers(upstream_edge_group)
        for lane_idx_left_most, lane_suffix in enumerate(list(range(edges_lane_numbers[0]))[::-1]):
            beginning_lane_id = edge_id + "_" + str(lane_suffix)
            lane_group = []
            lane_leftover = leftover_length
            for edge_idx, edge_lane_number in enumerate(edges_lane_numbers):
                lane_id = upstream_edge_group[edge_idx] + "_" + str(edge_lane_number - 1 - lane_idx_left_most)
                if lane_idx_left_most < edge_lane_number:
                    lane_group.append(lane_id)
                else:
                    # detection range is longer than this certain lane group
                    lane_leftover = 0
                    break

            single_edge_lane_groups[beginning_lane_id] = {"lane_group": lane_group, "leftover_length": lane_leftover}

        return single_edge_lane_groups

    def get_edges_upstream_lane_groups_with_length(self, edge_ids: list, length: float):
        upstream_lane_groups = {}
        for edge_id in edge_ids:
            upstream_lane_groups.update(self._get_edge_upstream_lane_groups_with_length(edge_id, length))

        return upstream_lane_groups

    def get_time(self):
        return self._flow_kernel.kernel_api.simulation.getTime()

    def get_detector_activation(self, det_id, activation_type):
        """
        Given the detector ID, returns the activation for the detector.
        Activation is based on the activation_type chosen: 'position', 'entry' or 'exit'.
        If 'position': detector is activated for all the timesteps for which vehicle is on detector.
        If 'entry': detector is only activated when vehicle enters the detector area.
        If 'exit': detector is only activated when vehicle exits the detector area.

        Args:
            det_id (str): Detector ID.
            activation_type (str): Can be 'position', 'entry' or 'exit'.

        Returns:
            int: 0 or 1. Detector activation.
        """
        POSITION, ENTRY, EXIT = DetectorActivation.POSITION, DetectorActivation.ENTRY, DetectorActivation.EXIT

        if activation_type == POSITION:
            return self.get_last_step_vehicle_count_over_detector(det_id)
        elif activation_type == ENTRY or activation_type == EXIT:
            veh_data = self.get_vehicle_data_over_detector(det_id)
            if not veh_data:
                return 0

            time = self.get_time()
            _, _, entry_time, exit_time, _ = next(iter(veh_data))

            if activation_type == ENTRY:
                return 1 if (time - entry_time) < 1 else 0
            else:
                return 1 if (time - exit_time) < 1 else 0
        else:
            raise ValueError(activation_type)

    def get_initial_rl_vehicle_ids(self):
        """ Get the initial rl vehicle ids from flow kernel """
        rl_veh_type_lst = []
        type_parameters = self._flow_kernel.vehicle.type_parameters
        rl_id_list_cp = []
        
        # Loop through the type_parameters and find the rl vehicle types
        for veh_type in type_parameters:
            if type_parameters[veh_type]['acceleration_controller'][0] == RLController:
                rl_veh_type_lst.append(veh_type)

        print(rl_veh_type_lst)
        # Push the rl_veh ids to self.rl_id_list_cp
        if type(self).__name__ == 'TraciKernel':
            for veh_id, veh_config in self._flow_kernel.vehicle._TraCIVehicle__vehicles.items():
                # Add it to self.rl_id_list_cp, the latter condition
                # is just double check
                print(veh_config)
                if veh_config['type'] in rl_veh_type_lst and\
                     veh_id not in rl_id_list_cp:
                    rl_id_list_cp.append(veh_id)
        
        return rl_id_list_cp

    def get_rl_vehicle_ids(self):
        return self._flow_kernel.vehicle.get_rl_ids()

    def get_vehicle_leader(self, veh_id):
        return self._flow_kernel.vehicle.get_leader(veh_id)

    def get_vehicle_follower(self, veh_id):
        return self._flow_kernel.vehicle.get_follower(veh_id)

    def get_vehicle_headway(self, veh_id):
        return self._flow_kernel.vehicle.get_headway(veh_id)
    
    def get_lane_leaders(self, veh_id):
        return self._flow_kernel.vehicle.get_lane_leaders(veh_id)

    def get_lane_followers(self, veh_id):
        return self._flow_kernel.vehicle.get_lane_followers(veh_id)

    def get_lane_leaders_speed(self, veh_id):
        return self._flow_kernel.vehicle.get_lane_leaders_speed(veh_id)
    
    def get_lane_headways(self, veh_id):
        return self._flow_kernel.vehicle.get_lane_headways(veh_id)
    
    def get_lane_tailways(self, veh_id):
        return self._flow_kernel.vehicle.get_lane_tailways(veh_id)

    def get_edge_list(self):
        return self._flow_kernel.network.get_edge_list()

    def get_edge_lane_mapping(self):
        return self._flow_kernel.network.get_edge_lane_mapping()
    
    def apply_acceleration(self, veh_id, acc):
        self._flow_kernel.vehicle.apply_acceleration(veh_id, acc=acc)
    
    def get_speed_limit_by_edge(self, edge):
        """ Return the speed limit of a given edge/junction """
        return self._flow_kernel.network.speed_limit(edge)

    def get_edge(self, veh_id, error=""):
        """Return the edge the specified vehicle is currently on.

        Parameters
        ----------
        veh_id : str or list of str
            vehicle id, or list of vehicle ids
        error : any, optional
            value that is returned if the vehicle is not found

        Returns
        -------
        str
        """
        return self._flow_kernel.vehicle.get_edge(veh_id, error=error)
