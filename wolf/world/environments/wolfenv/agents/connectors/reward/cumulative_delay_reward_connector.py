import numpy as np
from gym.spaces import Discrete, Box
import csv

from wolf.world.environments.wolfenv.agents.connectors.reward.reward_connector import \
    AccumulateSumRewardConnector, JointRewardConnector


class DifferenceInCumulativeDelayRewardConnector(AccumulateSumRewardConnector):
    def __init__(self, connectors_ids, kernel, stop_speed=2, detection_range=300):
        connectors = [
            _MonoNodeDifferenceInCumulativeDelayRewardConnector(
                connectors_ids=[node_id],
                stop_speed=stop_speed,
                detection_range=detection_range,
                kernel=kernel)
            for node_id in connectors_ids]

        super().__init__(connectors_ids=connectors_ids, connectors=connectors, kernel=kernel)


class _MonoNodeDifferenceInCumulativeDelayRewardConnector(JointRewardConnector):
    """
    Difference in cumulative delay reward function.

    Immediate reward is the difference in the cumulative delay to the last timestep.
    Cumulative delay only takes the vehicles still staying in the detection range into account.
    Once a vehicle leaves the intersection, there will be a drop down in "cumulative delay".
    """
    def __init__(self, connectors_ids, kernel, stop_speed=2, detection_range=300):
        super().__init__(connectors_ids=connectors_ids, reward_space=Box(-np.inf, 0, ()), kernel=kernel)
        self._t = 0
        self._node_id = connectors_ids[0]
        self._stop_speed = stop_speed
        self._detection_range = detection_range
        self._vehicle_delays = {}

        self._last_timestep_cumulative_delay = 0
        self._incoming_edges = self._kernel.get_traffic_light_incoming_edges(self._node_id)
        self._incoming_edge_groups = self._kernel.get_upstream_edge_groups_with_length(self._incoming_edges, self._detection_range)

    def a_compute(self):
        self._t += 1
        
        veh_ids = []
        for incoming_edge_group in self._incoming_edge_groups.values():
            veh_ids += self._kernel.get_vehicle_in_detection_range(incoming_edge_group["edge_group"], incoming_edge_group["leftover_length"])
        veh_speeds = self._kernel.get_vehicle_speed(veh_ids)

        for veh_id, veh_speed in zip(veh_ids, veh_speeds):
            if veh_speed < self._stop_speed:
                if veh_id in self._vehicle_delays:
                    self._vehicle_delays[veh_id] += 1
                else:
                    self._vehicle_delays[veh_id] = 1
        
        for veh_id in list(self._vehicle_delays.keys()):
            if veh_id not in veh_ids:
                del self._vehicle_delays[veh_id]
        
        cumulative_delay = sum(self._vehicle_delays.values())
        difference_in_cumulative_delay = cumulative_delay - self._last_timestep_cumulative_delay
        self._last_timestep_cumulative_delay = cumulative_delay

        return np.array(-difference_in_cumulative_delay)

    def reset(self):
        self._t = 0
        self._vehicle_delays = {}
        self._last_timestep_cumulative_delay = 0
