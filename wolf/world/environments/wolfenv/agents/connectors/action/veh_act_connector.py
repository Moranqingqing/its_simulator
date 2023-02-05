from copy import deepcopy
import logging

from wolf.world.environments.wolfenv.agents.connectors.action.action_connector import ActionConnector


import numpy as np
import gym
from gym.spaces import Discrete, Box

class VehActionConnector(ActionConnector):

    def __init__(self, connectors_ids, kernel=None):
        self.veh_id = connectors_ids[0] # Expect only one id
        self.reaction_time = 1
        # TODO: Find a way to import config
        self.max_decel = -3 # env_params.additional_params['max_decel']
        self.max_accel = 3 # env_params.additional_params['max_accel']
        
        lb = [-abs(self.max_decel)]
        ub = [self.max_accel]

        action_space = Box(np.array(lb), np.array(ub), dtype=np.float32)

        super().__init__(action_space=action_space, kernel=kernel)
        self._LOGGER = logging.getLogger(__name__)

    def a_compute(self, action):
        # TODO: Find a way to adapt the _apply_rl_actions in BottleneckAccel
        # TODO: Might need to clip the action
        # veh_speed = self._kernel.get_vehicle_speed(self.veh_id)

        # lead_veh = self._kernel.get_vehicle_leader(self.veh_id)
        # if lead_veh is not None:
        #     lead_veh_speed = self._kernel.get_vehicle_speed(lead_veh)

        #     abs_max_decel = abs(self.max_decel)

        #     d_safe = veh_speed * self.reaction_time + \
        #             (veh_speed)**2 / (2*abs_max_decel) - \
        #             (lead_veh_speed)**2 / (2*abs_max_decel)

        #     distance_headway = self._kernel.get_vehicle_headway(self.veh_id)

        #     if distance_headway < d_safe:
        #         print("Get here")
        #         action = [self.max_decel]
        action = action[0]
        
        if self.veh_id in self._kernel.get_rl_vehicle_ids():
            self._kernel.apply_acceleration(self.veh_id, action)
        else:
            self._LOGGER.debug(f"For debugging, vehicle {self.veh_id} is currently not in the network")

    def reset(self):
        # Simply Return, we will see what to reset
        return



class VehActionConnector_lc(ActionConnector):

    def __init__(self, connectors_ids, kernel=None):
        self.veh_id = connectors_ids[0] # Expect only one id
        self.reaction_time = 1
        # TODO: Find a way to import config
        self.max_decel = -3 # env_params.additional_params['max_decel']
        self.max_accel = 3 # env_params.additional_params['max_accel']
        
        action_space = Box(low=np.array([self.max_decel, -3]), high=np.array([self.max_accel, 3]), dtype=np.float32)

        super().__init__(action_space=action_space, kernel=kernel)
        self._LOGGER = logging.getLogger(__name__)

    def a_compute(self, action):
        action=np.array(action)
        acc = action[0]
        direction=int(action[1]/3)
        
        if self.veh_id in self._kernel.get_rl_vehicle_ids():
            lane_id = self._kernel.get_edge(self.veh_id)
            if lane_id=='4255':
                direction=0
            # print('lane change direction', direction, 'lane id',lane_id)## add hard constraint to make sure it will not change to wrong lane
            self._kernel.apply_acceleration(self.veh_id, acc)
            self._kernel.apply_lane_change(self.veh_id, direction)
        else:
            self._LOGGER.debug(f"For debugging, vehicle {self.veh_id} is currently not in the network")

    def reset(self):
        # Simply Return, we will see what to reset
        return
        