import numpy as np
from gym.spaces import Box, Tuple
import csv

from wolf.world.environments.wolfenv.agents.connectors.reward.reward_connector import JointRewardConnector, \
    AccumulateSumRewardConnector


class AverageDelayMeasurementConnector(AccumulateSumRewardConnector):
    """
    Average delay **measurement** function class. This is a **network level** measurement.
    Hence it is not dependent on the node_id. And it only consists of one mono_node connector.
    """
    def __init__(self, connectors_ids, kernel=None, stop_speed=2, save_data=False):
        connectors = [_MonoNodeAverageDelayMeasurementConnector(
            kernel=kernel,
            stop_speed=stop_speed,
            save_data=save_data
        )]

        super().__init__(connectors_ids=connectors_ids, connectors=connectors, kernel=kernel)


class _MonoNodeAverageDelayMeasurementConnector(JointRewardConnector):
    """
    Average delay measurement. Returns the difference in average delay between the current timestep
    and the previous timestep as only the cumulative value is accessible at the end of the evaluation.
    Only consider completed trips.
    """
    def __init__(self, kernel, stop_speed=2, save_data=False):
        super().__init__(connectors_ids=None, reward_space=Box(-np.inf, np.inf, ()), kernel=kernel)
        self._t = 0
        self._stop_speed = stop_speed
        self._save_data = save_data

        self._last_timestep_average_delay = 0
        self._vehicle_delays = {}
        self._complete_trip_delays = {}

    def a_compute(self):
        self._t += 1
        
        veh_ids = self._kernel._flow_kernel.vehicle.get_ids()
        veh_speeds = self._kernel.get_vehicle_speed(veh_ids)

        for veh_id, veh_speed in zip(veh_ids, veh_speeds):
            if veh_id in self._vehicle_delays:
                self._vehicle_delays[veh_id] += 1 if (veh_speed < self._stop_speed) else 0
            else:
                self._vehicle_delays[veh_id] = 1 if (veh_speed < self._stop_speed) else 0
        
        for veh_id in list(self._vehicle_delays.keys()):
            if veh_id not in veh_ids:
                self._complete_trip_delays[veh_id] = self._vehicle_delays[veh_id]
                del self._vehicle_delays[veh_id]
        
        if len(self._complete_trip_delays) != 0:
            average_delay = sum(self._complete_trip_delays.values()) / len(self._complete_trip_delays)
        else:
            average_delay = 0
        difference_in_average_delay = average_delay - self._last_timestep_average_delay
        self._last_timestep_average_delay = average_delay

        if self._save_data:
            self._save_vehicle_data_to_csv(self._t, self._complete_trip_delays)

        return np.array(-difference_in_average_delay)

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
        self._last_timestep_average_delay = 0
        self._vehicle_delays = {}
        self._complete_trip_delays = {}
