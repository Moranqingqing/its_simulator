import numpy as np
from gym.spaces import Box, Tuple
import csv

from wolf.world.environments.wolfenv.agents.connectors.reward.reward_connector import JointRewardConnector, \
    AccumulateSumRewardConnector


class AverageTravelTimeMeasurementConnector(AccumulateSumRewardConnector):
    """
    Average travel time **measurement** function class. This is a **network level** measurement.
    Hence it is not dependent on the node_id. And it only consists of one mono_node connector.
    """
    def __init__(self, connectors_ids, kernel=None, save_data=False, **kwargs):
        connectors = [_MonoNodeAverageTravelTimeMeasurementConnector(
            kernel=kernel,
            save_data=save_data
        )]

        super().__init__(connectors_ids=connectors_ids, connectors=connectors, kernel=kernel)


class _MonoNodeAverageTravelTimeMeasurementConnector(JointRewardConnector):
    """
    Average travel time measurement. Returns the difference in average travel time between the current
    timestep and the previous timestep as only the cumulative value is accessible at the end of the evaluation.
    Only consider completed trips.
    """
    def __init__(self, kernel, save_data=False):
        super().__init__(connectors_ids=None, reward_space=Box(-np.inf, np.inf, ()), kernel=kernel)
        self._t = 0
        self._save_data = save_data

        self._last_timestep_ATT = 0
        self._veh_start_times = {}
        self._veh_travel_times = {}

    def a_compute(self):
        self._t += 1
        departed_veh_ids = self._kernel._flow_kernel.vehicle.kernel_api.simulation.getDepartedIDList()
        arrived_veh_ids = self._kernel._flow_kernel.vehicle.kernel_api.simulation.getArrivedIDList()

        
        for veh_id in departed_veh_ids:
            self._veh_start_times[veh_id] = self._t
        
        for veh_id in arrived_veh_ids:
            if veh_id in self._veh_start_times:
                self._veh_travel_times[veh_id] = self._t - self._veh_start_times[veh_id]
                del self._veh_start_times[veh_id]

        if len(self._veh_travel_times) != 0:
            average_travel_time = sum(self._veh_travel_times.values()) / len(self._veh_travel_times)
        else:
            average_travel_time = 0

        difference_in_ATT = average_travel_time - self._last_timestep_ATT
        self._last_timestep_ATT = average_travel_time

        if self._save_data:
            self._save_vehicle_data_to_csv(self._t, self._veh_travel_times)

        return np.array(-difference_in_ATT)

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
        self._last_timestep_ATT = 0
        self._veh_start_times = {}
        self._veh_travel_times = {}
