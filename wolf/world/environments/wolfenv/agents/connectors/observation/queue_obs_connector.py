import numpy as np
import logging

from gym.spaces import Discrete, Dict
from collections import deque

from wolf.world.environments.wolfenv.agents.connectors.observation.obs_connector import \
    JointObsConnector

LOGGER = logging.getLogger(__name__)


class QueueObservationConnector(JointObsConnector):
    def __init__(
        self,
        connectors_ids,
        kernel=None,
        detection_range=300,
        stop_speed=2,
        bins=[1, 7, 14],
        embed_actionable=False,
        **kwargs
    ):
        self.connectors = [
            _MonoNodeQueueObsConnector(
                connectors_ids=[node_id],
                kernel=kernel,
                detection_range=detection_range,
                stop_speed=stop_speed,
                bins=bins,
                embed_actionable=embed_actionable
            )
            for node_id in connectors_ids
        ]
        observation_space = self.connectors[0].obs_space()
        super().__init__(
            connectors_ids=connectors_ids, observation_space=observation_space, kernel=kernel
        )

    def a_compute(self):
        obs = self.connectors[0].compute()

        return obs


class _MonoNodeQueueObsConnector(JointObsConnector):
    def __init__(
        self,
        connectors_ids,
        kernel=None,
        detection_range=300,
        stop_speed=3,
        bins=[1, 7, 14],
        embed_actionable=False
    ):
        self._node_id = connectors_ids[0]
        self._detection_range = detection_range
        self._stop_speed = stop_speed
        self._bins = bins
        self._embed_actionable = embed_actionable

        self._traffic_light = kernel.traffic_lights.get_traffic_light(self._node_id)
        self._n = self._traffic_light.len_green_phases()
        
        if self._embed_actionable:
            observation_space = Discrete((len(self._bins)+1) ** self._n * 2)
        else:
            obs_queue = Discrete((len(self._bins)+1) ** self._n)
            obs_actable = Discrete(2)
            observation_space = Dict({"queue": obs_queue, "actable": obs_actable})

        super().__init__(connectors_ids=connectors_ids, observation_space=observation_space, kernel=kernel)

        self._incoming_edges = kernel.get_traffic_light_incoming_edges(self._node_id)
        self._incoming_lanes = kernel.get_traffic_light_incoming_lanes(self._node_id)

        self._green_phases_lanes = self._match_green_phases_lanes()
        self._lanes_groups = kernel.get_edges_upstream_lane_groups_with_length(self._incoming_edges,
                self._detection_range)

        self._LOGGER = logging.getLogger(__name__)

    def _match_green_phases_lanes(self):
        green_phases_lanes = []

        green_phases_colors = self._traffic_light.get_green_phases_colors()
        movement_lane = self._kernel.get_traffic_light_movement_lane(self._node_id)

        for colors in green_phases_colors:
            green_phase_lanes = []
            for idx, color in enumerate(colors):
                if (color == "g" or color == "G") and idx in movement_lane:
                    green_phase_lanes.append(movement_lane[idx])
            
            green_phase_lanes = list(set(green_phase_lanes))
            green_phases_lanes.append(green_phase_lanes)

        return green_phases_lanes


    def _get_vehicles_in_detection_range(self, lane_group, leftover_length):
        # previous [:-1] lanes are all covered by the detection range
        veh_ids = self._kernel.get_vehicle_ids_by_lanes(lane_group[:-1])
        # the last lane may not be entirely covered by the detection range
        pending_veh_ids = self._kernel.get_vehicle_ids_by_lanes(lane_group[-1])

        if leftover_length >= 0:
            veh_ids += pending_veh_ids
        else:
            for veh in pending_veh_ids:
                pending_veh_pos = self._kernel.get_vehicle_position(veh)
                if pending_veh_pos > -leftover_length:
                    veh_ids.append(veh)

        return veh_ids

    def _bin_queue(self, queues):
        for idx, queue in enumerate(queues):
            state = len(self._bins)
            for i, bound in enumerate(self._bins):
                if queue <= bound:
                    state = i
                    break
            
            queues[idx] = state
        
        return queues

    def _discretize_state(self, queues):
        cardin = len(self._bins) + 1
        state = 0
        for queue in queues:
            state *= cardin
            state += queue

        return int(state)

    def a_compute(self):
        queues = []
        for idx in range(self._n):
            green_phase_lanes = self._green_phases_lanes[idx]
            queue = 0
            for incomging_lane_id in green_phase_lanes:
                lane_vehicles = self._get_vehicles_in_detection_range(
                    self._lanes_groups[incomging_lane_id]["lane_group"],
                    self._lanes_groups[incomging_lane_id]["leftover_length"],
                )
                veh_speeds = self._kernel.get_vehicle_speed(lane_vehicles)
                vehicle_count_below_stop_speed = len(
                    list(filter(lambda x: x < self._stop_speed, veh_speeds))
                )
                if vehicle_count_below_stop_speed > queue:
                    queue = vehicle_count_below_stop_speed

            queues.append(queue)

        queues = self._bin_queue(queues)
        obs_queue = self._discretize_state(queues)
        obs_actable = 1 if self._traffic_light.is_actionable() else 0
        # if "center0" in self._node_id:
        #     print(f"=== queue: {queues} ===== actable: {obs_actable} ===")

        state = obs_actable * int(self._observation_space.n/2) + obs_queue \
            if self._embed_actionable else {"queue": obs_queue, "actable": obs_actable}

        return state

