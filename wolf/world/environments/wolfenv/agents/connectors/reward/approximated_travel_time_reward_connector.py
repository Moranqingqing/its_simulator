import numpy as np
from gym.spaces import Box, Tuple
import copy
import csv
from collections import defaultdict

from wolf.world.environments.wolfenv.agents.connectors.reward.reward_connector import JointRewardConnector, \
    AccumulateSumRewardConnector


class ApproximatedTravelTimeRewardConnector(AccumulateSumRewardConnector):
    """
    Approximated travel time **reward** function class. Travel time is estimated by the detectors' countings.
    To see more details about the travel time's definition, please checkout travel_time_reward_connector.py
    """
    def __init__(self, connectors_ids, kernel=None, save_data=False):
        connectors = [
            _MonoNodeApproximatedTravelTimeRewardConnector(
                connectors_ids=[node_id],
                save_data=save_data,
                kernel=kernel)
            for node_id in connectors_ids]

        super().__init__(connectors_ids=connectors_ids, connectors=connectors, kernel=kernel)


class _MonoNodeApproximatedTravelTimeRewardConnector(JointRewardConnector):
    def __init__(self, connectors_ids, kernel, num_detector_group=2, save_data=False):
        super().__init__(connectors_ids=connectors_ids, reward_space=Box(-np.inf, 0, ()), kernel=kernel)
        self._t = 0
        self._node_id = connectors_ids[0]
        self._num_detector_group = num_detector_group
        self._save_data = save_data

        self._incoming_edges = self._kernel.get_traffic_light_incoming_edges(self._node_id)
        self._edge_detector_groups = self._kernel.get_intersection_edge_detector_groups(self._node_id, self._incoming_edges, self._num_detector_group)

        self._veh_travel_times = {}
        for edge in self._incoming_edges:
            self._veh_travel_times[edge] = []

    def a_compute(self):
        self._t += 1

        self._update_travel_times()
        travel_times = [sum(x) for x in self._veh_travel_times.values()]
        travel_time = sum(travel_times)

        # if self._save_data:
        #     self._save_vehicle_data_to_csv(self._t, self._veh_travel_times)
        
        return np.array(-travel_time)

    # def _save_vehicle_data_to_csv(self, timestep, veh_travel_times, filename='travel_times.csv'):
    #     self.write_to_csv(timestep, veh_travel_times, filename)

    # def write_to_csv(self, t, veh_travel_times, filename):
    #     veh_travel_times = veh_travel_times.copy()
    #     with open(filename, 'w') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=['t'] + list(veh_travel_times.keys()))
    #         veh_travel_times['t'] = t
    #         writer.writerow(veh_travel_times)

    def _update_travel_times(self):
        for edge in self._incoming_edges:
            # new vehicle comes from upstream detectors
            incoming_veh_count = 0
            for detector in self._edge_detector_groups[edge][1]:
                vehicle_data = self._kernel.get_vehicle_data_over_detector(detector)
                for veh in vehicle_data:
                    if veh[3] != -1.:
                        incoming_veh_count += 1
            self._veh_travel_times[edge] += [0] * incoming_veh_count

            # vehicle left from downstream detectors
            outgoing_veh_count = 0
            for detector in self._edge_detector_groups[edge][0]:
                vehicle_data = self._kernel.get_vehicle_data_over_detector(detector)
                for veh in vehicle_data:
                    if veh[3] != -1.:
                        outgoing_veh_count += 1
            self._veh_travel_times[edge] = self._veh_travel_times[edge][outgoing_veh_count:]

            # add 1s for all vehicles in the detection range
            self._veh_travel_times[edge] = [x + 1 for x in self._veh_travel_times[edge]]

    def reset(self):
        self._t = 0
        self._veh_travel_times = {}
        for edge in self._incoming_edges:
            self._veh_travel_times[edge] = []
