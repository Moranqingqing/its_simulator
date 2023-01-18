import numpy as np
import logging

from gym.spaces.box import Box

from wolf.world.environments.wolfenv.agents.connectors.observation.obs_connector import ObsConnector, \
    JointObsConnector
from wolf.world.environments.wolfenv.kernels.wolf_kernel import WolfKernel
from wolf.world.environments.wolfenv.phase import GREEN, GREEN_PRIORITY

class DTSEConnector(JointObsConnector):

    def __init__(self, connectors_ids, max_speed, obs_params, kernel=None, **kwargs):
        """

        Parameters
        ----------
        @ node_id: str
            the intersection's id
        @ max_speed: float
            used for normalization
        @ obs_params: dict
            parameters corresponding to the observation space
        @ kernel: flow.core.kernel
            kernel to the simulator
        """
        #######################################################
        node_id = connectors_ids[0]
        node_connections = kernel.get_node_connections(node_id)
        # hard coding the order, to be fixed!!!!!!!!!!!!!!!!
        self._incoming_edges = list(set([c['from'] for c in node_connections]))
        order = {'lef': 0, 'top': 1, 'rig': 2, 'bot': 3}
        self._incoming_edges.sort(key=lambda x:order[x[0:3]])

        self._incoming_lanes = list(
            set([f"{c['from']}_{c['fromLane']}" for c in node_connections])
        )
        self._incoming_lanes, _ = WolfKernel._sort_lanes(self._incoming_edges, self._incoming_lanes)
        self._lane_lengths = [kernel._flow_kernel.network.edge_length(x[:-2]) for x in self._incoming_lanes]
        #######################################################

        self._num_lanes = len(self._incoming_lanes)
        self._max_speed = max_speed
        self._num_cell = obs_params["num_cell"]
        self._cell_length = obs_params["cell_length"]
        self._detection_length = self._num_cell * self._cell_length

        super().__init__(connectors_ids=connectors_ids, observation_space=Box(0, 1, (self._num_lanes, self._num_cell, 2)), kernel=kernel, **kwargs)

        self._LOGGER = logging.getLogger(__name__)

        self._dtse = None


    def a_compute(self):
        """
        Get the newest wolfenv state, build the dtse tensor, and return it

        0                          300
        |                           |       position from the simulator
        -----------------------------
        -----------------------------
                           |        |       position for the dtse
                           0       100

        Parameters
        ----------
        None

        Returns
        ----------
        @ _dtse: np.array
            the DTSE tensor
        """
        POSITION = 0
        VELOCITY = 1

        dtse = np.zeros((self._num_lanes, self._num_cell, 2))

        for i in range(self._num_lanes):
            lane_length = self._lane_lengths[i]
            assert lane_length >= self._detection_length

            vehicle_ids = self._kernel._flow_kernel.vehicle.get_ids_by_lane(self._incoming_lanes[i])
            veh_velocitys = self._kernel.get_vehicle_speed(vehicle_ids)

            n_veh_per_cell = np.zeros(self._num_cell)
            cumulated_speed_per_cell = np.zeros(self._num_cell)

            for j, veh_id in enumerate(vehicle_ids):
                # calculate the position from the downstream end of the lane
                position = self._kernel._flow_kernel.vehicle.get_position(veh_id) + self._detection_length - lane_length
                if position < self._detection_length:
                    cell_idx = int(position // self._cell_length)
                    n_veh_per_cell[cell_idx] += 1
                    cumulated_speed_per_cell[cell_idx] += veh_velocitys[j]
            
            for cell_idx in range(self._num_cell):
                if n_veh_per_cell[cell_idx] != 0:
                    dtse[i, cell_idx, POSITION] = 1
                    dtse[i, cell_idx, VELOCITY] = cumulated_speed_per_cell[cell_idx] / n_veh_per_cell[cell_idx]

        dtse[:, :, VELOCITY] = dtse[:, :, VELOCITY] / self._max_speed   # normalize the speed matrix

        self._dtse = dtse

        return self._dtse
