from wolf.world.environments.traffic.agents.connectors.done.done_connector import DoneConnector
import numpy as np


class VehDoneConnector(DoneConnector):
    def __init__(self, veh_ids, kernel=None):
        self.veh_ids = veh_ids[0]
        super().__init__(kernel=kernel)

    def compute(self):
        # TODO: Check if this really necessary
        if self.veh_ids not in self._kernel._flow_kernel.vehicle.get_rl_ids():
            return 1
        return 0