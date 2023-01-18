from collections import OrderedDict

import numpy as np
import logging

from gym.spaces import Tuple, Discrete
from gym.spaces.box import Box
from gym.spaces.dict import Dict
from ray.rllib.utils.debug import summarize

from wolf.world.environments.wolfenv.agents.connectors.connector import AgentListener
from wolf.world.environments.wolfenv.agents.connectors.observation.obs_connector import \
    JointObsConnector
from wolf.utils.enums import DetectorActivation

import logging

from wolf.world.environments.wolfenv.agents.wolf_agent import WolfAgent

LOGGER = logging.getLogger(__name__)


class TDTSEStateSpace(Dict):
    def __init__(
            self,
            num_lanes,
            num_history,
            num_detector_group,
            phase_channel,
            num_intersection=1,
            use_progression=False,
            max_avail_actions=None,
            has_avail_actions=False,
    ):
        self.num_lanes = num_lanes
        self.num_history = num_history
        self.num_detector_group = num_detector_group
        self.phase_channel = phase_channel
        self.num_intersection = num_intersection
        self._num_channels = (num_detector_group + 1) if phase_channel else num_detector_group
        dim = (num_intersection, num_lanes, num_history, self._num_channels)

        extras = {}
        if has_avail_actions:
            extras['action_mask'] = Box(0, 1, shape=(max_avail_actions,))

        if use_progression:
            extras['max_progression'] = Box(0, 1, (num_intersection,), np.float32)
            extras['min_progression'] = Box(0, 1, (num_intersection,), np.float32)

        extras['actionable'] = Box(0, 1, (num_intersection,))

        super().__init__(tdtse=Box(0, 1, dim, np.uint8), **extras)


class TDTSERawStateSpace(Box):
    def __init__(
            self,
            num_lanes,
            num_history,
            num_detector_group,
            phase_channel,
            num_intersection=1,
            use_progression=False,
            max_avail_actions=None,
            has_avail_actions=False,
    ):
        self.num_lanes = num_lanes
        self.num_history = num_history
        self.num_detector_group = num_detector_group
        self.phase_channel = phase_channel
        self.num_intersection = num_intersection
        self._num_channels = (num_detector_group + 1) if phase_channel else num_detector_group
        max_avail_actions = max_avail_actions if has_avail_actions else 0
        dim = (num_intersection * num_lanes * num_history * self._num_channels + (
            2 if use_progression else 0) + max_avail_actions,)
        super().__init__(0, 1, dim, np.uint8)


class TDTSEConnector(JointObsConnector):
    def __init__(
            self,
            connectors_ids,
            obs_params,
            phase_channel=True,
            kernel=None,
            raw_obs=False,
            detector_activation=DetectorActivation.POSITION,
            use_progression=False,
            action_connector=None,
            **kwargs
    ):
        self.raw_obs = raw_obs
        self.detector_activation = detector_activation
        self.use_progression = use_progression
        action_space = action_connector._action_space
        self.avail_actions_method = getattr(action_connector, 'avail_actions_method', None)

        if raw_obs:
            LOGGER.warning('raw_obs is only usefull for QMIX.')

        self.connectors = [
            _MonoNodeTDTSEConnector(
                connectors_ids=[node_id],
                obs_params=obs_params,
                phase_channel=phase_channel,
                kernel=kernel,
                raw_obs=raw_obs,
                detector_activation=detector_activation,
                use_progression=use_progression,
                action_space=action_space,
                avail_actions_method=self.avail_actions_method
            )
            for node_id in connectors_ids
        ]
        conn = self.connectors[0]
        cls = TDTSERawStateSpace if raw_obs else TDTSEStateSpace
        obs_space = cls(
            num_lanes=conn._num_lanes,
            num_history=conn._num_history,
            num_detector_group=conn._num_detector_group,
            phase_channel=conn._phase_channel,
            num_intersection=len(connectors_ids),
            use_progression=use_progression,
            max_avail_actions=conn._max_avail_actions,
            has_avail_actions=bool(conn._get_avail_actions),
        )
        super().__init__(connectors_ids=connectors_ids, observation_space=obs_space, kernel=kernel, **kwargs)

    def attach_agent(self,wolf_agent: WolfAgent):

        # @parth: TODO

        super().attach_agent(wolf_agent)



    def a_compute(self):
        obs = [conn.compute() for conn in self.connectors]
        if self.raw_obs:
            obs = np.concatenate(obs)
        else:
            tdtse = np.concatenate([o["tdtse"] for o in obs])
            if len(tdtse.shape) != 4:
                raise Exception(f'TDTSE rank should be 4. TDTSE provided: {summarize(tdtse)}')

            extras = {}
            if self.avail_actions_method:
                extras['action_mask'] = np.concatenate([o["action_mask"] for o in obs])

            if self.use_progression:
                extras['max_progression'] = np.concatenate([o["max_progression"] for o in obs])
                extras['min_progression'] = np.concatenate([o["min_progression"] for o in obs])

            extras['actionable'] = np.concatenate([o["actionable"] for o in obs])

            obs = dict(tdtse=tdtse, **extras)
        return obs

    def reset(self):
        for conn in self.connectors:
            conn.reset()

    def reset(self):
        for conn in self.connectors:
            conn.reset()


class _MonoNodeTDTSEConnector(JointObsConnector):
    def __init__(
            self,
            connectors_ids,
            obs_params,
            phase_channel=True,
            kernel=None,
            raw_obs=False,
            detector_activation=DetectorActivation.POSITION,
            use_progression=False,
            action_space=None,
            avail_actions_method=None,
            **kwargs
    ):
        """[summary]

        Args:
            connectors_ids (list): controlled intersection's id, should be a list with len=1
            obs_params (dict): observation connector parameters
            phase_channel (bool, optional): use phase channel or not. Defaults to True.
            kernel (wolf.wolfenv.kernel, optional): wolf's kernel. Defaults to None.
            raw_obs (bool, optional): use raw observation space or not. Defaults to False.
            detector_activation (str, optional): Can be: 'position', 'entry' or 'exit'. Defaults to 'position'.
                For more checkout self._kernel.get_detector_activation and DetectorActivation enum.
            use_progression (bool, optional): use progression or not. Defaults to False.
            action_space ([type], optional): [description]. Defaults to None.
            avail_actions_method ([type], optional): [description]. Defaults to None.
        """
        self.raw_obs = raw_obs
        self.detector_activation = detector_activation
        self.use_progression = use_progression
        self._node_id = connectors_ids[0]

        if "num_detector_group" in obs_params:
            # for real network, these two params shouldn't exist together
            self._num_detector_group = obs_params["num_detector_group"]
        elif "detector_position" in obs_params:
            # for grid network
            self._num_detector_group = len(obs_params["detector_position"])
        else:
            raise KeyError("Should identify detectors.")

        self._incoming_lanes = kernel.get_traffic_light_incoming_lanes(self._node_id)
        self._lane_detectors = kernel.get_ordered_detector_groups(self._node_id, self._num_detector_group)

        # inclues the pseudo-lanes for extra upstream detectors
        self._num_lanes = len(self._lane_detectors)
        self._num_history = obs_params["num_history"]
        self._phase_channel = phase_channel
        self._current_phase = None
        self._traffic_light = kernel.traffic_lights.get_traffic_light(self._node_id)

        assert isinstance(action_space, Discrete)
        self._max_avail_actions = action_space.n
        self._get_avail_actions = getattr(self._traffic_light, str(avail_actions_method), None)

        cls = TDTSERawStateSpace if raw_obs else TDTSEStateSpace
        observation_space = cls(
            num_lanes=self._num_lanes,
            num_history=self._num_history,
            num_detector_group=self._num_detector_group,
            phase_channel=self._phase_channel,
            num_intersection=1,
            use_progression=use_progression,
            max_avail_actions=self._max_avail_actions,
            has_avail_actions=bool(self._get_avail_actions),
        )

        super().__init__(connectors_ids=connectors_ids, observation_space=observation_space, kernel=kernel, **kwargs)

        self._LOGGER = logging.getLogger(__name__)
        self._lane_movements = self._kernel.get_traffic_light_lane_movements(self._node_id)

        self._tdtse_phase = np.zeros((self._num_lanes, self._num_history, 1))
        self._tdtse_detector = np.zeros((self._num_lanes, self._num_history, self._num_detector_group))

    def a_compute(self):
        """
        Get the newest wolfenv state, update the tdtse tensor, and return it.

        Returns:
            np.array: tdtse tensor
        """
        self._current_phase = self._traffic_light.get_phase()
        phase = []  # the last column for the phase channel
        detector = []  # the last columns for the detector channels
        for lane in self._lane_detectors:
            # get the colors for all the movements corresponding to this lane if this is a true lane
            # pseudo-lanes always get red
            colors = [self._current_phase.colors[x] for x in self._lane_movements[lane]] if \
                lane in self._incoming_lanes else ['r']
            phase.append(1 if 'g' in colors or 'G' in colors else 0)

            detector_entry = [
                1 if x is not None and self._kernel.get_detector_activation(x["id"],
                                                                            self.detector_activation) > 0 else 0
                for x in self._lane_detectors[lane].values()
            ]
            detector.append(detector_entry)

        phase = np.array(phase, dtype=np.uint8)
        detector = np.array(detector, dtype=np.uint8)
        phase = phase.reshape((self._num_lanes, 1, 1))
        detector = detector.reshape((self._num_lanes, 1, self._num_detector_group))

        # add the newest columns at the end, remove the first columns at the front
        self._tdtse_phase = np.concatenate((self._tdtse_phase, phase), axis=1)[:, 1:, :]
        self._tdtse_detector = np.concatenate((self._tdtse_detector, detector), axis=1)[:, 1:, :]

        if self._phase_channel:
            tdtse = np.concatenate((self._tdtse_detector, self._tdtse_phase), axis=2)
        else:
            tdtse = self._tdtse_detector

        tdtse = np.expand_dims(tdtse, axis=0)  # adding "intersection" dimension
        if self.use_progression:
            max_progression = np.array([self._traffic_light.max_progress()])
            min_progression = np.array([self._traffic_light.min_progress()])

        if self.raw_obs:
            if self.use_progression:
                d = np.concatenate([tdtse.flatten(), min_progression, max_progression])
            else:
                d = tdtse.flatten()
        else:
            extras = {}
            if self._get_avail_actions:
                extras['action_mask'] = self._get_avail_actions()

            if self.use_progression:
                extras['max_progression'] = max_progression
                extras['min_progression'] = min_progression

            extras['actionable'] = [self._traffic_light.is_actionable()]

            d = OrderedDict(tdtse=tdtse, **extras)

        return d

    def reset(self):
        self._tdtse_phase = np.zeros((self._num_lanes, self._num_history, 1))
        self._tdtse_detector = np.zeros((self._num_lanes, self._num_history, self._num_detector_group))
        self._current_phase = None
        self._traffic_light.reset()
