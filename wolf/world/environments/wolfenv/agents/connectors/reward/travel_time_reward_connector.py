import numpy as np
from gym.spaces import Box, Tuple
import csv

from wolf.world.environments.wolfenv.agents.connectors.reward.reward_connector import JointRewardConnector, \
    AccumulateSumRewardConnector


class TravelTimeRewardConnector(AccumulateSumRewardConnector):
    """
    Travel time **reward** function class. To see the average travel time measurement,
    please checkout average_travel_time_measurement_connector.py.
    
    Immediate reward is the cumulative travel time of the set of vehicles stay inside the
    detection range of an intersection. Once a vehicle left the intersection, there will be
    a drop down in the "travel time".
    """
    def __init__(self, connectors_ids, kernel=None, save_data=False):
        connectors = [
            _MonoNodeTravelTimeRewardConnector(
                connectors_ids=[node_id],
                save_data=save_data,
                kernel=kernel)
            for node_id in connectors_ids]

        super().__init__(connectors_ids=connectors_ids, connectors=connectors, kernel=kernel)


class _MonoNodeTravelTimeRewardConnector(JointRewardConnector):
    def __init__(self, connectors_ids, kernel, detection_range=300, save_data=False):
        super().__init__(connectors_ids=connectors_ids, reward_space=Box(-np.inf, 0, ()), kernel=kernel)
        self._t = 0
        self._node_id = connectors_ids[0]
        self._detection_range = detection_range
        self._save_data = save_data

        self._veh_travel_times = {}
        self._incoming_edges = self._kernel.get_traffic_light_incoming_edges(self._node_id)
        self._incoming_edge_groups = self._kernel.get_upstream_edge_groups_with_length(self._incoming_edges, self._detection_range)

    def a_compute(self):
        self._t += 1

        veh_ids = []
        for incoming_edge_group in self._incoming_edge_groups.values():
            veh_ids += self._get_vehicle_in_detection_range(incoming_edge_group["edge_group"], incoming_edge_group["leftover_length"])

        # if the vehicle appears for the first time, record the start time.
        for veh_id in veh_ids:
            if veh_id in self._veh_travel_times:
                self._veh_travel_times[veh_id] += 1
            else:
                self._veh_travel_times[veh_id] = 1
        
        # a recorded vehicle no longer exists in the detection range means it leaves the intersection.
        # remove it from the recording list.
        for veh_id in list(self._veh_travel_times.keys()):
            if veh_id not in veh_ids:
                del self._veh_travel_times[veh_id]

        travel_time = sum(self._veh_travel_times.values())

        if self._save_data:
            self._save_vehicle_data_to_csv(self._t, self._veh_travel_times)

        return np.array(-travel_time)

    def _save_vehicle_data_to_csv(self, timestep, veh_travel_times, filename='travel_times.csv'):
        self.write_to_csv(timestep, veh_travel_times, filename)

    def write_to_csv(self, t, veh_travel_times, filename):
        veh_travel_times = veh_travel_times.copy()
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['t'] + list(veh_travel_times.keys()))
            veh_travel_times['t'] = t
            writer.writerow(veh_travel_times)

    def reset(self):
        self._t = 0
        self._veh_travel_times = {}
