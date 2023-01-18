import numpy as np
from gym.spaces import Discrete, Box
import csv

from wolf.world.environments.wolfenv.agents.connectors.reward.reward_connector import (
    JointRewardConnector,
    AccumulateSumRewardConnector)


class QueueRewardConnector(AccumulateSumRewardConnector):
    def __init__(self, connectors_ids, kernel, stop_speed=2, detection_range=300, save_data=False):
        connectors = [
            _MonoNodeQueueRewardConnector(
                connectors_ids=[node_id],
                stop_speed=stop_speed,
                detection_range=detection_range,
                save_data=save_data,
                kernel=kernel)
            for node_id in connectors_ids]

        super().__init__(connectors_ids=connectors_ids, connectors=connectors, kernel=kernel)


class _MonoNodeQueueRewardConnector(JointRewardConnector):
    """"""

    # TODO: take into account if queue is longer than the incoming edge.
    # TODO: check if units of speed are the same

    def __init__(self, connectors_ids, kernel, stop_speed=2, detection_range=300, save_data=False):
        super().__init__(connectors_ids=connectors_ids, reward_space=Box(-np.inf, 0, ()), kernel=kernel)
        self._t = 0
        self._node_id = connectors_ids[0]
        self._stop_speed = stop_speed
        self._detection_range = detection_range
        self._save_data = save_data

        self._incoming_edges = self._kernel.get_traffic_light_incoming_edges(self._node_id)
        self._incoming_edge_groups = self._kernel.get_upstream_edge_groups_with_length(self._incoming_edges, self._detection_range)

    def _get_vehicles_in_detection_range(self, edge_group, leftover_length):
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
        veh_ids = self._kernel.get_vehicle_ids_by_edge(edge_group[:-1])
        # the last edge may not be entirely covered by  the detection range
        pending_veh_ids = self._kernel.get_vehicle_ids_by_edge(edge_group[-1])

        if leftover_length >= 0:
            veh_ids += pending_veh_ids
        else:
            for veh in pending_veh_ids:
                pending_veh_pos = self._kernel.get_vehicle_position(veh)
                if pending_veh_pos > -leftover_length:
                    veh_ids.append(veh)
        
        return veh_ids


    def a_compute(self):
        self._t += 1
        
        veh_ids = []
        for incoming_edge_group in self._incoming_edge_groups.values():
            veh_ids += self._get_vehicles_in_detection_range(incoming_edge_group["edge_group"], incoming_edge_group["leftover_length"])

        veh_speeds = self._kernel.get_vehicle_speed(veh_ids)

        vehicle_count_below_stop_speed = len(
            list(filter(lambda x: x < self._stop_speed, veh_speeds))
        )

        if self._save_data:
            self._save_vehicle_data_to_csv(self._t, vehicle_count_below_stop_speed)
            
        return np.array(-vehicle_count_below_stop_speed)

    def _save_vehicle_data_to_csv(self, timestep, queue_count, filename='queue_data.csv'):
        self.write_to_csv(timestep, queue_count, filename)

    def write_to_csv(self, t, queue_count, filename):
        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['t', 'queue_count'])
            writer.writerow({'t': t, 'queue_count': queue_count})

    def reset(self):
        self._t = 0
