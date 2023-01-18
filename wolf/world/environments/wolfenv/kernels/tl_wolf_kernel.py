from abc import ABC, abstractmethod
from itertools import cycle, dropwhile
import numpy as np
import logging
import copy
from wolf.world.environments.wolfenv.phase import Phase, PhaseType


class TrafficLightsWolfKernel:
    def __init__(self, kernel, tl_params):
        self.logger = logging.getLogger(__name__)
        self.kernel = kernel
        self.traffic_lights = {}
        controlled_nodes_ids = self.kernel.get_controlled_nodes_ids()
        from wolf.utils.configuration.registry import R
        if "ALL" in tl_params:
            all_tl_params = tl_params["ALL"]
            for node_id in controlled_nodes_ids:
                node_tl_params = self._update_tl_params_for_real_network(all_tl_params, node_id)
                self.traffic_lights[node_id] = R.traffic_light_class(node_tl_params["name"])(
                    node_id=node_id,
                    kernel=kernel,
                    **node_tl_params["params"])
        # else:
        for node_id, params in tl_params.items():
            # allowing individual tls' params overwrite the "ALL" params
            if node_id in controlled_nodes_ids:
                self.traffic_lights[node_id] = R.traffic_light_class(params["name"])(
                    node_id=node_id,
                    kernel=kernel,
                    **params["params"])
            else:
                if node_id is not "ALL":
                    raise KeyError(f"Node {node_id} is not a controlled_node.")

    def len(self):
        return len(self.traffic_lights)

    def reset(self):
        self.logger.debug("Resetting {} traffic lights".format(
            len(self.traffic_lights.values())))
        for tl in self.traffic_lights.values():
            tl.reset()

    def get_traffic_light(self, node_id):
        return self.traffic_lights[node_id]

    def _update_tl_params_for_real_network(self, config_tl_params, node_id):
        """
        Complete the tl_params for each intersection if there is "default_constraints" exists in the "params".
        Default constraints will be applied to Phases (read from the network) with unspecified constraints.

        Args:
            config_tl_params (dict): the tl_params from the config
            node_id (str): traffic light's ID

        Returns:
            dict: complemented tl_params
        """
        if "default_constraints" not in config_tl_params["params"]:
            return config_tl_params
        default_constraints_params = config_tl_params["params"]["default_constraints"]
        
        tl_params = copy.deepcopy(config_tl_params)
        tl_params["params"]["phases"] = []
        phases = self.kernel.get_traffic_light_logic(node_id)
        phases_type = PhaseType.get_phases_type(phases)

        for phase, phase_type in zip(phases, phases_type):
            if phase_type == PhaseType.GREEN:
                tl_params["params"]["phases"].append({
                    "colors": phase["colors"],
                    "min_time": default_constraints_params["min_time"],
                    "max_time": default_constraints_params["max_time"],
                })
            else:
                tl_params["params"]["phases"].append({
                    "colors": phase["colors"],
                    "min_time": phase["duration"],
                    "max_time": phase["duration"],
                })
        
        for phase_params in config_tl_params["params"].get("phases", []):
            to_replace = self._find_phase_by_color(tl_params["params"]["phases"], phase_params["colors"])
            assert to_replace is not None, \
                "Phase \"{}\" doesn't exist for tl \"{}\". Please check your tl_params again.".format(phase_params["colors"], node_id)
            to_replace["min_time"] = phase_params["min_time"]
            to_replace["max_time"] = phase_params["max_time"]

        # in the rest of the code, don't need this flag anymore
        tl_params["params"].pop("default_constraints")

        return tl_params
    
    @staticmethod
    def _find_phase_by_color(phases, colors):
        for phase in phases:
            if isinstance(phase, dict):
                if phase["colors"] == colors:
                    return phase
            else:
                if phase.colors == colors:
                    return phase
        return None


class TrafficLight(ABC):
    def __init__(self, node_id, kernel):
        """
        :param node_id:
        :param kernel:
        :param logic: cycle_based of time_based and params
        """
        self.kernel = kernel
        self._node_id = node_id
        self._current_phase = None

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def get_phase(self):
        return self._current_phase


class AvailActionsMethod:
    """
    Enum for the avail action methods that an action-connector can choose from.
    """
    PHASE_SELECT = 'get_phase_select_avail_actions'
    EXTEND_CHANGE = 'get_extend_change_avail_actions'


class SecondBasedTrafficLight(TrafficLight):
    def __init__(self, phases, node_id, initialization, kernel):
        TrafficLight.__init__(self, node_id, kernel)
        self._phases_list = phases
        self.initialization = initialization

        for i, phase in enumerate(self._phases_list):
            if isinstance(phase, dict):
                self._phases_list[i] = Phase.from_dict(phase)

        self._phase_types = PhaseType.get_phases_type(self._phases_list)
        self._green_phase_groups = self._get_green_phase_groups()
        self._green_phases_colors = [g[0].colors for g in self._green_phase_groups]
        self._num_phases_per_green_group = [len(g) for g in self._green_phase_groups]
        self._num_green_phases = len(self._green_phase_groups)

        self._current_phase_t = 0
        self._current_phase_idx = None
        self._current_green_phase_group_idx = None
        self._delayed_action = None     # for variable phasing, store the last available action
                                        # before it could be executed.

        self.logger = logging.getLogger("TrafficLight {}".format(node_id))
        self.logger.debug("phases_list:\n{}".format(
            "\n".join([str(phase) for phase in self._phases_list])))

    def _get_green_phase_groups(self):
        """
        Get green phase groups for variable phasing traffic lights.
        Green phase group: a list of phases begining with a green phase and followed by
            in-between phases

        Returns:
            list[list[wolf.phase.Phase]]: the list of green phase groups

            >>> self._get_green_phase_groups([GREEN, IN_BETWEEN, IN_BETWEEN, GREEN, IN_BETWEEN])
            >>> [[Phase[0], Phase[1], Phase[2]],
                [Phase[3], Phase[4]]]
        """
        green_phase_groups = []
        assert self._phase_types[0] == PhaseType.GREEN
        for idx, phase_type in enumerate(self._phase_types):
            if phase_type == PhaseType.GREEN:
                green_phase_groups.append([self._phases_list[idx]])
            else:
                green_phase_groups[-1].append(self._phases_list[idx])

        return green_phase_groups

    def _idx_2_green_idx(self, phase_idx):
        """
        Get the index of the current green phase group.

        Args:
            phase_idx (int): the actual phase index

        Returns:
            int: the green phase index

            >>> self._green_phase_groups = [[P0, P1, P2], [P3, P4]]
            >>> self._num_phases_per_green_group = [3, 2]
            >>> self._idx_2_green_idx(3)
            >>> 1
        """
        for idx, num_phase in enumerate(self._num_phases_per_green_group):
            phase_idx -= num_phase
            if phase_idx < 0:
                return idx
    
    def _green_idx_2_idx(self, green_phase_group_idx):
        """
        Get a green phase's actual phase index from its green phase index.

        Args:
            green_phase_group_idx (int): the green phase index

        Returns:
            int: the actual phase index

            >>> self._green_phase_groups = [[P0, P1, P2], [P3, P4]]
            >>> self._num_phases_per_green_group = [3, 2]
            >>> self._green_idx_2_idx(1)
            >>> 3
        """
        return sum(self._num_phases_per_green_group[0: green_phase_group_idx])
    
    def get_green_phases_colors(self):
        return self._green_phases_colors

    def len_phases(self):
        return len(self._phases_list)

    def len_green_phases(self):
        return self._num_green_phases

    def reset(self):
        if self.initialization == 'random':
            rand_idx = np.random.randint(0, len(self._phases_list))
            self._current_phase = self._phases_list[rand_idx]
            self._current_phase_idx = rand_idx
            self._current_green_phase_group_idx = self._idx_2_green_idx(rand_idx)
            self.logger.debug("logger: Resetting, setting random initial phase: {}".format(str(self._current_phase)))
        elif self.initialization == 'fixed':
            self._current_phase = self._phases_list[0]
            self._current_phase_idx = 0
            self._current_green_phase_group_idx = 0
            self.logger.debug("logger: Resetting, setting fixed initial phase: {}".format(str(self._current_phase)))
        else:
            raise Exception(f'Unexpected initialization string: {self.initialization}')

        self._current_phase_t = 0
        self._delayed_action = None
        self.kernel.set_traffic_light_phase(
            node_id=self._node_id, current_phase=self._current_phase)

    def max_progress(self):
        return min(self._current_phase_t/(self._current_phase.max_time - 1), 1)

    def min_progress(self):
        return min(self._current_phase_t/(self._current_phase.min_time - 1), 1)

    def select_phase_by_index(self, next_phase_index):
        if self.is_action_valid(next_phase_index):
            actual_next_phase_index = self._green_idx_2_idx(next_phase_index)

            if self._current_phase_idx == actual_next_phase_index:
                self.extend()
            else:
                self.change()
        else:
            raise ValueError(f'Unexpected action: {next_phase_index} recieved.')

    def switch_to_phase(self, next_green_phase_idx):
        """
        Handle the raw action from the agent.

        When the traffic light is in an actionable state and receives a valid raw
        action different from the current green phase, it will store the action
        in self._delayed_action and try to execute it.

        To execute such an action, the traffic light have to exit the current green
        phase and go through all subsequencing in-between phases. Hence, the action
        execution will be postponed. And the self._delayed_action is necessary.

        When the traffic light has a delayed action, it will ignore all newly coming
        raw actions and always try to execute the stored one. Once the delayed action
        has been succesfully executed, the self._delayed_action will be cleared by
        self.switch(), which also lets the traffic light be available for new actions.
        """
        if self._delayed_action is not None:
            self.switch()
        else:
            if self._phase_types[self._current_phase_idx] == PhaseType.GREEN:
                if self._current_green_phase_group_idx == next_green_phase_idx:
                    self.extend()
                else:
                    if self._current_phase_t >= (self._current_phase.min_time - 1):
                        self._delayed_action = next_green_phase_idx
                        self.switch()
                    else:
                        self.extend()
            else:
                self.extend()

    def is_action_valid(self, action):
        return bool(self._prev_avail_actions[action])

    def extend(self):
        self.logger.debug("Extend. Current phase: {}".format(str(self._current_phase)))
        if self._current_phase_t < (self._current_phase.max_time - 1):
            self._extend_phase()
        else:
            self._next_phase()

    def change(self):
        self.logger.debug("Change. Current phase: {}".format(str(self._current_phase)))
        if self._current_phase_t >= (self._current_phase.min_time - 1):
            self._next_phase()
        else:
            self._extend_phase()

    def switch(self):
        """
        Lower-level phase switching method.

        Lead the traffic light go throught in-between phases like common fixed order TLs.
        Switch to the designated green phase when and only when the next time-step would be
        the beginning of a green phase.
        """
        self.logger.debug("Switch. Current phase: {}".format(str(self._current_phase)))
        if self._current_phase_t >= (self._current_phase.min_time - 1):
            if self._phase_types[(self._current_phase_idx + 1) % len(self._phases_list)] == \
                    PhaseType.GREEN:
                self._switch_phase()
            else:
                self._next_phase()
        else:
            self._extend_phase()
        
    def noop(self):
        self.logger.debug("No-op. Current phase: {}".format(str(self._current_phase)))
        # assert self._current_phase_t < self._current_phase.max_time
        if self._current_phase_t >= (self._current_phase.max_time - 1):
            # action should be change here not noop. if noop: debug further.
            print('current_t', self._current_phase_t, 'max_time', self._current_phase.max_time)
        self._extend_phase()

    def get_phase_select_avail_actions(self):
        # TODO: give better variable names below. eg: curr_or_next_green_phase.
        avail_actions = self._num_green_phases * [0]
        in_between_phases = False

        curr_or_next_green_phase = self.get_current_or_next_green_phase()
        curr_index = self._green_phases_colors.index(curr_or_next_green_phase.colors)
        next_index = (curr_index + 1) % self._num_green_phases

        if curr_or_next_green_phase != self._current_phase:
            in_between_phases = True

        min_progress = self.min_progress()
        max_progress = self.max_progress()

        if min_progress < 1 or in_between_phases:
            avail_actions[curr_index] = 1
        elif max_progress < 1:
            avail_actions[curr_index] = 1
            avail_actions[next_index] = 1
        else:
            avail_actions[next_index] = 1

        self._prev_avail_actions = avail_actions
        return avail_actions

    def get_current_or_next_green_phase(self):
        # if current_phase is green, return current phase.
        if self._current_phase.colors in self._green_phases_colors:
            return self._current_phase

        # TODO (parth): use self._phases_list to find current or next green phase.
        phases = cycle(self._phases_list)
        for _ in range(self._current_phase_idx + 1):
            phase = next(phases)

        while True:
            phase = next(phases)
            if phase.colors in self._green_phases_colors:
                return phase
        
    def get_extend_change_avail_actions(self):
        only_noop = [0, 0, 1]
        only_change = [0, 1, 0]
        extend_change = [1, 1, 0]

        min_progression = self.min_progress()
        max_progression = self.max_progress()

        if min_progression < 1:
            return only_noop
        if max_progression == 1:
            return only_change
        return extend_change

    def is_actionable(self):
        """
        Returns True if traffic light is actionable, returns False if it is not.
        """
        extend_change = [1, 1, 0]
        ec_avail_actions = self.get_extend_change_avail_actions()
        is_actionable = (ec_avail_actions == extend_change)
        return is_actionable

    def _extend_phase(self):
        self._current_phase_t += 1

    def _next_phase(self):
        self._current_phase_idx = (self._current_phase_idx + 1) % len(self._phases_list)
        self._current_phase = self._phases_list[self._current_phase_idx]
        self._current_green_phase_group_idx = self._idx_2_green_idx(self._current_phase_idx)
        self._current_phase_t = 0

        self.kernel.set_traffic_light_phase(
            node_id=self._node_id, current_phase=self._current_phase)

    def _switch_phase(self):
        """
        Switch the traffic light to the designated green phase. Clear the stored
        postponed action (self._delayed_action).
        """
        self._current_phase_idx = self._green_idx_2_idx(self._delayed_action)
        self._current_phase = self._phases_list[self._current_phase_idx]
        self._current_green_phase_group_idx = self._delayed_action
        self._current_phase_t = 0
        self._delayed_action = None

        self.kernel.set_traffic_light_phase(
            node_id=self._node_id, current_phase=self._current_phase)


class CycleBasedTrafficLight(TrafficLight):
    def __init__(self, phases, node_id, kernel):
        TrafficLight.__init__(self, node_id, kernel)
        self._phases_list = phases
        for i in range(len(self._phases_list)):
            phase = self._phases_list[i]
            if isinstance(phase, dict):
                self._phases_list[i] = Phase.from_dict(phase)
        self._phases = None
        self._current_phase_t = 0
        self.logger = logging.getLogger("TrafficLight {}".format(node_id))
        self.logger.debug("phases_list:\n{}".format(
            "\n".join([str(phase) for phase in self._phases_list])))

        self._phase_types = PhaseType.get_phases_type(self._phases_list)
        if self._phase_types[0] != PhaseType.GREEN:
            # TODO: should be able to handle the rest situations
            raise Exception(
                f"The first phase of a full cycle should be a green phase, rather than a {self._phase_types[0]} phase")

        self._major_inter_phases = []
        for i, phase in enumerate(self._phases_list):
            if self._phase_types[i] == PhaseType.GREEN:
                self._major_inter_phases.append({"major": phase, "inter": []})
            else:
                self._major_inter_phases[-1]["inter"].append(phase)

        self._current_cycle_t = 0
        self.cycle_time = None

    def num_green_phases(self):
        return len(self._major_inter_phases)

    def len_phases(self):
        return len(self._phases_list)

    def reset(self):

        self._phases = cycle(self._phases_list)
        rand_idx = np.random.randint(0, len(self._phases_list))
        for i in range(rand_idx + 1):
            self._current_phase = next(self._phases)
        self.logger.debug("logger: Resetting, setting random initial phase: {}".format(
            str(self._current_phase)))
        self._current_phase_t = 0
        self.kernel.set_traffic_light_phase(
            node_id=self._node_id, current_phase=self._current_phase)

    @staticmethod
    def _cycle_add_phase(cycle_phases, phase, cycle_time, phase_len=0):
        import math
        duration = math.ceil((phase.max_time - phase.min_time)
                             * phase_len + phase.min_time)  # TODO: or round?
        cycle_phases.append({'duration': duration, 'state': phase.colors})

        return cycle_time + duration

    def set_cycle(self, action_cycle):
        # action can only be executed after the last full cycle finished
        if self.cycle_time != None and self._current_cycle_t < (self.cycle_time - 1):
            self._current_cycle_t += 1
        # calculate/re-calculate the new cycle time at the beginning of the epoch or
        # the new cycle
        else:
            self.cycle_time = 0
            self._current_cycle_t = 0

            cycle_phases = []
            for idx in action_cycle["order"]:
                phase_set = self._major_inter_phases[idx]
                self.cycle_time = self._cycle_add_phase(
                    cycle_phases, phase_set["major"], self.cycle_time, action_cycle["length"][idx])
                for inter_phase in phase_set["inter"]:
                    self.cycle_time = self._cycle_add_phase(
                        cycle_phases, inter_phase, self.cycle_time)

            self.kernel.set_traffic_light_logic(
                node_id=self._node_id, cycle_logic=cycle_phases)


class FixedOrderCycleBasedTrafficLight(CycleBasedTrafficLight):
    def set_cycle(self, action_cycle):
        # action can only be executed after the last full cycle finished
        if self.cycle_time != None and self._current_cycle_t < (self.cycle_time - 1):
            self._current_cycle_t += 1
        # calculate/re-calculate the new cycle time at the beginning of the epoch or
        # the new cycle
        else:
            self.cycle_time = 0
            self._current_cycle_t = 0

            cycle_phases = []
            for phase_set, length in zip(self._major_inter_phases, action_cycle):
                self.cycle_time = self._cycle_add_phase(
                    cycle_phases, phase_set["major"], self.cycle_time, length)
                for inter_phase in phase_set["inter"]:
                    self.cycle_time = self._cycle_add_phase(
                        cycle_phases, inter_phase, self.cycle_time)

            self.kernel.set_traffic_light_logic(
                node_id=self._node_id, cycle_logic=cycle_phases)


class PhaseSplitTrafficLight(CycleBasedTrafficLight):
    def __init__(self, phases, cycle_time, node_id, kernel):
        CycleBasedTrafficLight.__init__(self, phases, node_id, kernel)
        self.cycle_time = cycle_time
        self._current_cycle_t = self.cycle_time
        self.__cycle_extensible_green_time = cycle_time

        for phase_set in self._major_inter_phases:
            self.__cycle_extensible_green_time -= phase_set["major"].min_time
            for phase in phase_set["inter"]:
                if phase.min_time == phase.max_time:
                    self.__cycle_extensible_green_time -= phase.min_time
                else:
                    raise Exception(
                        "For the intermediate phases, the min_time should be equal to the max_time")

        if self.__cycle_extensible_green_time < 0:
            raise Exception(
                "Invalid cycle_time. The cycle_time should be larger than the sum of all min_times")

    @staticmethod
    def _cycle_add_phase(cycle_phases, phase, extension=0):
        duration = phase.min_time + extension
        cycle_phases.append({'duration': duration, 'state': phase.colors})

    def set_cycle(self, action_cycle):
        # action can only be executed after the last full cycle finished
        if self._current_cycle_t < (self.cycle_time - 1):
            self._current_cycle_t += 1
        # calculate/re-calculate the new cycle time at the beginning of the epoch or
        # the new cycle
        else:
            self._current_cycle_t = 0

            cycle_phases = []
            # TODO: refine the extension strategy
            extensions = [round(
                a * (self.__cycle_extensible_green_time / sum(action_cycle))) for a in action_cycle]
            extensions[-1] = self.__cycle_extensible_green_time - \
                sum(extensions[:-1])
            assert min(extensions) >= 0

            for phase_set, extension in zip(self._major_inter_phases, extensions):
                self._cycle_add_phase(cycle_phases, phase_set["major"], extension)
                for inter_phase in phase_set["inter"]:
                    self._cycle_add_phase(cycle_phases, inter_phase)

            self.kernel.set_traffic_light_logic(
                node_id=self._node_id, cycle_logic=cycle_phases)
            
