from collections import OrderedDict

import numpy as np
import logging

from gym.spaces import Tuple, Discrete
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from ray.rllib.utils.debug import summarize

from wolf.world.environments.wolfenv.agents.connectors.observation.obs_connector import\
    VehConnector
from wolf.world.environments.wolfenv.agents.wolf_agent import WolfAgent

import logging

LOGGER = logging.getLogger(__name__)


class CarFollowingStateSpace(Dict):
    def __init__(self, num_vehs, max_lanes, scaling):
        # the dimension depends on the model we implement
        dim = 5
        super().__init__(veh_obs=Box(low=-np.inf, high=np.inf, shape=(dim, ), dtype=np.float32))

class CarFollowingRawStateSpace(Box):
    def __init__(self, num_vehs, max_lanes, scaling):
        # the dimension depends on the model we implement
        dim = 5
        super().__init__(-np.inf, np.inf, shape=(dim, ), dtype=np.float32)


# ==== CAR FOLLOWING ====
class CarFollowingConnector(VehConnector):
    def __init__(
        self,
        connectors_ids,
        kernel=None,
        raw_obs=True,
        action_connector=None,
    ):
        # TODO: Check should we use only one agent, in another word,
        # Observation should be concantanated
        self.raw_obs = raw_obs
        action_space = action_connector._action_space
        self.avail_actions_method = getattr(action_connector, 'avail_actions_method', None)

        network_k = kernel._flow_kernel.network
        edge_list = network_k.get_edge_list()
        self._num_edges = len(edge_list)
        self._MAX_LANES = max( network_k.num_lanes(edge) for edge in edge_list)
        self._scaling = kernel._flow_network.net_params.additional_params['scaling']

        self.connectors = [
            _MONOVehCarFollowingConnector(
                connectors_ids=[veh_id],
                max_lanes=self._MAX_LANES,
                num_edges=self._num_edges,
                scaling=self._scaling,
                kernel=kernel,
                raw_obs=raw_obs
            )
            for veh_id in connectors_ids
        ]

        # Generate obs_space for parent class
        cls = CarFollowingRawStateSpace if raw_obs else CarFollowingStateSpace
        obs_space = cls(
            num_vehs=len(connectors_ids),
            max_lanes=self._MAX_LANES,
            scaling=self._scaling
        )
        super().__init__(connectors_ids=connectors_ids, observation_space=obs_space, kernel=kernel)

    def a_compute(self):
        # obs = [conn.compute() for conn in self.connectors]
        obs = [conn.compute(self.agent) for conn in self.connectors]
        if self.raw_obs:
            obs = np.concatenate(obs)
        else:
            veh_obs = np.concatenate([o['veh_obs'] for o in obs])

            obs = dict(veh_obs=veh_obs)
        return obs

class _MONOVehCarFollowingConnector(VehConnector):
    def __init__(
        self,
        connectors_ids,
        max_lanes,
        num_edges,
        scaling,
        kernel=None,
        raw_obs=True
    ):
        self.raw_obs = raw_obs
        self._veh_id = connectors_ids[0]

        #TODO: Find a way to get the scaling and MAX_LANES of the network/env
        self._MAX_LANES = max_lanes
        self._num_edges = num_edges
        self._scaling = scaling


        cls = CarFollowingRawStateSpace if raw_obs else CarFollowingStateSpace

        observation_space = cls(
            num_vehs=1,
            max_lanes=max_lanes,
            scaling=scaling
        )

        super().__init__(connectors_ids=connectors_ids, observation_space=observation_space, kernel=kernel)

        self._LOGGER = logging.getLogger(__name__)

    def a_compute(self, agent: WolfAgent):
        # Calculate the vehicle speed
        veh_speed = self._kernel.get_vehicle_speed(self._veh_id)

        # Get the leading vehicle
        lead_veh_id = self._kernel.get_vehicle_leader(self._veh_id)

        if lead_veh_id is None:
            # This is a temp fix for the vehicle in the front all exit
            # but the get_vehicle_leader won't return the vehicle just
            # enter the road
            #lead_veh_id = self._kernel.get_lane_leaders(self._veh_id)[0]
            #lead_veh_speed = self._kernel.get_vehicle_speed(lead_veh_id)
            #distance_headway = self._kernel.get_lane_headways(self._veh_id)[0]
            lead_veh_speed = 0
            distance_headway = 1000
        else:
            lead_veh_speed = self._kernel.get_vehicle_speed(lead_veh_id)
            # Get the headway
            distance_headway = self._kernel.get_vehicle_headway(self._veh_id)

        # Invert the invalid distance_headway, current default err value is -1001
        if distance_headway < 0:
            distance_headway = 1000

        # Calculate the relative speed
        rel_speed = max(lead_veh_speed, 0) - veh_speed

        # last acceleration
        prev_accel = 0
        if len(agent.actions) > 1:
            prev_accel = agent.actions[-2][0]

        # speed_limit
        curr_edge_idx = self._kernel.get_edge(self._veh_id)
        speed_limit = self._kernel.get_speed_limit_by_edge(curr_edge_idx)

        # Initialize as an array
        obs = np.array([veh_speed, rel_speed, distance_headway, speed_limit, prev_accel])
        if self.raw_obs:
            d = obs
        else:
            d = OrderedDict(veh_obs=obs)
        return d

    def compute(self, agent: WolfAgent):
        observation = self.a_compute(agent)

        if not self._observation_space.contains(observation):
            raise ValueError("Observation\n{}\noutside expected space\n{}"
                             .format(summarize(observation), self._observation_space))

        return observation



class BCMStateSpace(Dict):
    def __init__(self, num_vehs):
        # the dimension depends on the model we implement
        dim = 7
        super().__init__(veh_obs=Box(low=-np.inf, high=np.inf, shape=(dim, ), dtype=np.float32))
class BCMRawStateSpace(Box):
    def __init__(self, num_vehs):
        # the dimension depends on the model we implement
        dim = 7*num_vehs
        super().__init__(-np.inf, np.inf, shape=(dim, ), dtype=np.float32)


# ==== BCM ====
class BCMObsConnector(VehConnector):

    def __init__(
        self,
        connectors_ids,
        # phase_channel=True,
        kernel=None,
        raw_obs=True,
        # detector_activation=DetectorActivation.POSITION,
        # use_progression=False,
        action_connector=None,
    ):
        # TODO: Check should we use only one agent, in another word,
        # Observation should be concantanated
        self.raw_obs = raw_obs
        action_space = action_connector._action_space
        self.avail_actions_method = getattr(action_connector, 'avail_actions_method', None)

        self.connectors = [
            _MONOBCMObsConnector(
                connectors_ids=[veh_id],
                kernel=kernel,
                raw_obs=raw_obs
            )
            for veh_id in connectors_ids
        ]

        # Generate obs_space for parent class
        cls = BCMRawStateSpace if raw_obs else BCMStateSpace
        obs_space = cls(
            num_vehs=len(connectors_ids),
        )
        super().__init__(connectors_ids=connectors_ids, observation_space=obs_space, kernel=kernel)

    def a_compute(self):
        # obs = [conn.compute() for conn in self.connectors]
        obs = [conn.compute(self.agent) for conn in self.connectors]
        if self.raw_obs:
            obs = np.concatenate(obs)
        else:
            veh_obs = np.concatenate([o['veh_obs'] for o in obs])

            obs = dict(veh_obs=veh_obs)
        return obs



class _MONOBCMObsConnector(VehConnector):
    def __init__(
        self,
        connectors_ids,
        kernel=None,
        raw_obs=True
    ):
        self.raw_obs = raw_obs
        self._veh_id = connectors_ids[0]

        cls = BCMRawStateSpace if raw_obs else BCMStateSpace

        observation_space = cls(
            num_vehs=1,)

        super().__init__(connectors_ids=connectors_ids, observation_space=observation_space, kernel=kernel)

        self._LOGGER = logging.getLogger(__name__)

    def a_compute(self, agent: WolfAgent):

        # Calculate the vehicle speed
        veh_speed = self._kernel.get_vehicle_speed(self._veh_id)

        # Get the leading vehicle
        lead_veh_id = self._kernel.get_vehicle_leader(self._veh_id)

        if lead_veh_id is None:
            # This is a temp fix for the vehicle in the front all exit
            # but the get_vehicle_leader won't return the vehicle just
            # enter the road
            lead_veh_id = self._kernel.get_lane_leaders(self._veh_id)[0]
            lead_veh_speed = self._kernel.get_vehicle_speed(lead_veh_id)
            distance_headway = self._kernel.get_lane_headways(self._veh_id)[0]
        else:
            lead_veh_speed = self._kernel.get_vehicle_speed(lead_veh_id)
            # Get the headway
            distance_headway = self._kernel.get_vehicle_headway(self._veh_id)

        # Invert the invalid distance_headway, current default err value is -1001
        if distance_headway < 0:
            distance_headway = 1000

        # Calculate the relative speed
        rel_speed = max(lead_veh_speed, 0) - veh_speed

        # Get the follower
        follow_veh_id = self._kernel.get_vehicle_follower(self._veh_id)

        if follow_veh_id is None:
            follow_veh_id = self._kernel.get_lane_followers(self._veh_id)[0]
            follow_veh_speed = self._kernel.get_vehicle_speed(follow_veh_id)
            follow_distance_headway = self._kernel.get_lane_tailways(self._veh_id)[0]
        else:
            follow_veh_speed = self._kernel.get_vehicle_speed(follow_veh_id)
            follow_distance_headway = self._kernel.get_vehicle_headway(follow_veh_id)

        # Invert the invalid distance_headway, current default err value is -1001
        if follow_distance_headway < 0:
            follow_distance_headway = 1000
        
        # Calculate the relative speed of the current vehicles and the follower
        follow_rel_speed = max(follow_veh_speed, 0) - veh_speed

        # last acceleration
        prev_accel = 0
        if len(agent.actions) > 1:
            prev_accel = agent.actions[-2][0]

        # speed_limit
        curr_edge_idx = self._kernel.get_edge(self._veh_id)
        speed_limit = self._kernel.get_speed_limit_by_edge(curr_edge_idx)
        
        # Initialize as an array
        obs = np.array([veh_speed, rel_speed, distance_headway, follow_rel_speed, follow_distance_headway, speed_limit, prev_accel])
        if self.raw_obs:
            d = obs
        else:
            d = OrderedDict(veh_obs=obs)
        return d

    def compute(self, agent: WolfAgent):
        observation = self.a_compute(agent)

        if not self._observation_space.contains(observation):
            raise ValueError("Observation\n{}\noutside expected space\n{}"
                             .format(summarize(observation), self._observation_space))

        return observation
