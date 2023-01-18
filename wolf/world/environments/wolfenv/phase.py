from enum import Enum, IntEnum
from math import inf
from lxml import etree
import copy

# ===================================================================================
# SUMO Signals
# https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html

# Red: Vehicles must stop
# Yellow: Vehicles will start to decelerate if far away from the junction,
#         Otherwise vehicles pass
# Green (no priority): Vehicle may pass if no vehicle is on a higher-priority link
#                      Vehicles always decelerate on approach
# Green (priority): Vehicles may pass the junction
# Green (right-turn arrow) : Vehicles may pass if no vehicle is on a higher-priority link
#                            Vehicles always stop before passing
#                            Only generated for junction type 'traffic_light_right_on_red'
# Red+Yellow (orange) : Upcoming green phase, but vehicles may not drive yet
# Off (blinking) : Vehicles have to yield
# Off (no signal) : Vehicles have the right of way
#

RED = 'r'
YELLOW = 'y'
GREEN = 'g'
GREEN_PRIORITY = 'G'
GREEN_RIGHT_TURN = 's'
RED_YELLOW = 'u'
OFF_BLINKING = 'o'
OFF_NO_SIGNAL = 'O'

GREEN_SIGNALS = (GREEN, GREEN_PRIORITY, GREEN_RIGHT_TURN)
RED_SIGNALS = (RED, RED_YELLOW)

# ==================================================================================
# Aimsun Signals
#
# Flashing green  : Acts as green. Only for display purposes
# Flashing yellow : Vehicles apply giveway (yield) behaviour at the end of a section
#                   before taking the turn
# Flashing red    : Vehicles apply stop behaviour at the end of a section
#                   before taking the turn
#

AIMSUN_SIGNALS = {
    'RED': 0,
    'YELLOW': 2,
    'GREEN': 1,
    'FLASHING_GREEN_AS_GREEN': 3,
    'FLASHING_RED_AS_RED': 4,
    'FLASHING_YELLOW_AS_YELLOW': 5,
    'OFF': 6,
    'FLASHING_YELLOW_AS_YIELD': 7,
    'YELLOW_AS_GREEN': 8,
    'FLASHING_RED_AS_STOP': 9,
}

AIMSUN_GREEN_SIGNALS = (1, 3, 8)
AIMSUN_RED_SIGNALS = (0, 4)

AIMSUN_TO_WOLF_SIGNAL_MAPPING = {
    0: 'r',
    1: 'G',
    2: 'y',
    3: 'G',
    4: 'r',
    5: 'y',
    6: 'O',
    7: 'o',
    8: 'G',
    9: None,
}

WOLF_TO_AIMSUN_SIGNAL_MAPPING = {
    'r': 0,
    'y': 2,
    'g': 1,
    'G': 1,
    's': 1,
    'u': 0,
    'o': 7,
    '0': 6,
}

def set_index(obj, idx):
    """
    Sets the idx attribute of the passed object.

    Arguments
    ---------
        obj : Any
            The object for which the index will be set
        idx : Int
            The index to set. Should be a nonnegative integer
    """
    assert isinstance(idx, int) and idx >= 0
    obj.idx = idx

def make_index_setter(obj):
    """
    Creates an index setter method for the passed object

    Arguments
    ---------
        obj : Any

    Returns
    -------
        function int -> None
            Index setter for the object
    """
    return lambda idx: set_index(obj, idx)

def assign_indices(cltn):
    """
    For a passed iterable 'cltn', sets an 'idx' attribute for each element of
    the iterated sequence (the index is the index of the element in the iterated
    sequence).

    Arguments
    ---------
        cltn : Iterable of Any
            Iterable collection of objects
    """
    for idx, elt in enumerate(cltn):
        elt.set_index(idx)

# ==============================================================================================
# =================================== Phase class ==============================================

class Phase:
    def __init__(self, colors, min_time=0., max_time=inf, idx=None):
        if isinstance(colors, list):
            self.colors = "".join(colors)
        elif isinstance(colors, str):
            self.colors = colors
        else:
            raise TypeError("colors should be str or list")
        self.min_time = min_time
        self.max_time = max_time
        self.idx = idx
        self.set_index = make_index_setter(self)

    def __str__(self):
        str = "Phase(colors=\"{}\", min={}, max={})".format(self.colors,self.min_time,self.max_time)
        if hasattr(self, 'idx'): str = "{}. {}".format(self.idx, str)
        return str

    @staticmethod
    def from_dict(dict):
        return Phase(colors=dict["colors"],
                     min_time=float(dict["min_time"]),
                     max_time=float(dict["max_time"]))


class Turn:
    """
    A turning motion is an ordered pair
        (collection of incoming lanes, collection of outgoing lanes)

    In SUMO, a turning motion (link) typically contains a single
    incoming lane and a single outgoing lane

    In Aimsun, a turning motion typically contains more than one lane
    on both the incoming and the outgoing edge.

    Constructor parameters
    ----------------------
        in_edge, out_edge : Int or String
            Ids of the incoming and outgoing edges of the turn,
            respectively
        in_lanes : Tuple (Int, Int), optional
            The index of the leftmost lane and the index of the
            rightmost lane of the turn on the incoming edge, resp.
        out_lanes : Tuple (Int, Int), optional
            Similar to above, but for the outgoing edge
        idx: Int, optional
            The index of the turn in a collection of turns
    """
    def __init__(self,
                 in_edge,
                 out_edge,
                 in_lanes=None,
                 out_lanes=None,
                 idx=None):
        self.in_edge = in_edge
        self.out_edge = out_edge
        self.in_lanes = in_lanes
        self.out_lanes = out_lanes
        self.idx = idx
        self.set_index = make_index_setter(self)

    def set_in_lanes(self, in_laneL, in_laneR):
        """
        Parameters
        ----------
            in_laneL: int
                Index of the leftmost in-lane
            in_laneR: int
                Index of rightmost in-lane
        """
        assert all((isinstance(in_laneL, int),
                    isinstance(in_laneR, int),
                    in_laneL <= in_laneR))
        self.in_lanes = (in_laneL, in_laneR)

    def set_out_lanes(self, out_laneL, out_laneR):
        """
        Parameters
        ----------
            out_laneL: int
                Index of leftmost out-lane
            out_laneR: int
                Index of rightmost out-lane
        """
        assert all((isinstance(out_laneL, int),
                    isinstance(out_laneR, int),
                    out_laneL <= out_laneR))
        self.out_lanes = (out_laneL, out_laneR)

    def __eq__(self, other):
        """ Checks equality of two Turns """
        return all((self.in_edge == other.in_edge,
                    self.out_edge == other.out_edge,
                    (self.in_lanes is None) or (self.in_lanes == other.in_lanes),
                    (self.out_lanes is None) or (self.out_lanes == other.out_lanes)))

    def __neq__(self, other):
        return not self.eq(other)


class SignalGroup:
    """
    A signal group is a collection of turns, each of which receives
    the same signal.

    SUMO forgoes the organizational step of using signal groups
    (although it is possible to achieve the effect by assigning multiple links
     the same link index).

    Constructor parameters
    ----------------------
        turns : iterable of Turn
            The turns in the signal group
        idx : Int, optional
            The index of the signal group in a collection of signal groups

    Precondition
    ------------
        The passed turns should be a part of an enumerated collection of turns.
        As a consequence, each turn will have an 'idx' attribute. These indices
        will be recorded by the signal group instead of the turns themselves, for
        efficiency.
    """
    def __init__(self, *turns, idx=None):
        assert all(isinstance(turn, Turn) for turn in turns)
        assert all(hasattr(turn, 'idx') for turn in turns), \
               '[SignalGroup] The turns should be part of an enumeration of all turns in the node ' \
               '(the "assign_indices" function may be useful).'
        self.turn_indices = [turn.idx for turn in turns]
        self.idx = idx
        self.set_index = make_index_setter(self)

    @staticmethod
    def from_turn_indices(turn_indices, idx=None):
        """ Alternative constructor for defining the signal group from turn indices """
        sg = SignalGroup(idx=idx)
        sg.turn_indices = turn_indices
        return sg


class AimsunPhase(Phase):
    """
    In Aimsun, a phase is defined by a collection of signal groups that get
    the green light in the phase.

    Therefore, the STATE of the phase is generalized to mean a string like
    'GGGrrGr', where each letter encodes the signal for a particular signal group.
    In SUMO, each signal group may consist of a single turn (link), but in Aimsun
    the signal groups are typically larger.

    Constructor parameters
    ----------------------
        colors : String
            The state representation of the phase
        min_time, max_time : Floats, optional
            The minimum and maximum green times for the phase
        yellow_green_to_red : Float, optional
            The yellow green -> red time in the phase. If None, use the node value
        yellow_red_to_green : Float, optional
            The yellow red -> green time in the phase. If None, use the node value
        idx : Int, optional
            The index of the phase in a collection of phases (program logic)
        interphase : Boolean
            Whether the phase is an interphase

    Precondition
    ------------
        The passed signal groups should be part of an enumerated collection of signal groups.
        The state representation is with respect to this fixed ordering of the signal groups.
    """
    def __init__(self,
                 colors,
                 min_time=0.,
                 max_time=inf,
                 yellow_green_to_red=None,
                 yellow_red_to_green=None,
                 idx=None,
                 interphase=False):
        super().__init__(colors, min_time, max_time)
        self.green_sg_indices = [idx for idx in range(len(colors)) if colors[idx] in GREEN_SIGNALS]
        self.yellow_green_to_red = yellow_green_to_red
        self.yellow_red_to_green = yellow_red_to_green
        self.idx = idx
        self.interphase = interphase

    @staticmethod
    def from_signal_groups(*green_signal_groups,
                            num_sgs,
                            min_time=0.,
                            max_time=inf,
                            yellow_green_to_red=None,
                            yellow_red_to_green=None,
                            idx=None,
                            interphase=False):
        """
        Alternative constructor, directly from a collection of SignalGroup classes
        (the signal groups that get a green light in the phase are passed).

        Only the indices of the SignalGroups will be recorded. The SignalGroups
        should have been enumerated.

        New constructor parameters
        --------------------------
            *green_signal_groups : iterable of SignalGroup
                The signal groups that get a green light in the phase
            num_sgs : Int
                The total number of signal groups (needed for creating state strings)
        """
        assert all(isinstance(sg, SignalGroup) for sg in green_signal_groups)
        assert all(hasattr(sg, 'idx') for sg in green_signal_groups), \
               '[AimsunPhase] The signal groups should be part of an enumerated collection of ' \
               'signal groups (the "assign_indices" function may be useful).'
        green_sg_indices = [sg.idx for sg in green_signal_groups]
        colors = [GREEN_PRIORITY if idx in green_sg_indices else RED for idx in range(num_sgs)]
        return AimsunPhase(colors, min_time, max_time,
                           yellow_green_to_red, yellow_red_to_green, idx, interphase)

    @staticmethod
    def from_sg_indices(green_sg_indices,
                        num_sgs,
                        min_time=0.,
                        max_time=inf,
                        yellow_green_to_red=None,
                        yellow_red_to_green=None,
                        idx=None,
                        interphase=False):
        """ Alternative constructor for defining the phase from the signal group indices """
        colors = [GREEN_PRIORITY if idx in green_sg_indices else RED for idx in range(num_sgs)]
        return AimsunPhase(colors, min_time, max_time,
                           yellow_green_to_red, yellow_red_to_green, idx, interphase)

    def set_yellow_time_green_to_red(self, yellow_time):
        """
        Sets the duration of the yellow time from green to red in the phase.
        If not set, defaults to the node value (if the latter is set).
        """
        assert isinstance(yellow_time, (int, float))
        self.yellow_green_to_red = yellow_time

    def set_yellow_time_red_to_green(self, yellow_time):
        """
        Sets the duration of the yellow time from red to green in the phase.
        If not set, defaults to the node value (if the latter is set).
        """
        assert isinstance(yellow_time, (int, float))
        self.yellow_red_to_green = yellow_time

    def set_interphase(self, is_):
        """ Sets whether or not the phase is an interphase. The argument is_ is a Boolean """
        self.interphase = is_


class AimsunProgramLogic:
    """
    Contains the information necessary to set up a complete program
    for a traffic light node (see also the GKControlJunction class in Aimsun).

    The program logic is a sequence (list) of AimsunPhases. To have sufficient
    information to reconstruct the GKControlJunction, the AimsunProgramLogic
    also requires the enumerated collections of Turns and SignalGroups for
    the traffic light node.

    When the traffic light node is controlled by Aimsun (and not an external
    agent like Wolf), the attributes in the aimsun_controller_params dictionary
    are used.

    WARNING: Unless all yellow times are set to 0, Aimsun will place the yellow time
    at the beginning of the following phase. This means that the list of states
    in the ProgramLogic is incomplete if the traffic light is controlled by Aimsun.
    For a completed list of states, please use the 'get_completed_phase_list' method.

    If the traffic light node is controlled by Wolf, the list of phases in the
    ProgramLogic will be went through exactly, without additional phases (using
    the set_state method of the traffic light kernel).

    Constructor parameters
    ----------------------
        node_id : int
            The id of the traffic light node in Aimsun
        turns : list of Turn
            The signalled turning motions
        signal_groups : list of SignalGroup
            The signal groups that the turns are organized into
        phases : list of AimsunPhase
            The sequence of phases for the ProgramLogic
        aimsun_controller_params : Dictionary, optional
            'yellow_green_to_red' : float
                Default node yellow green -> red time
            'yellow_red_to_green' : float
                Default node yellow red -> green time
            'red_percentage': int
                Percentage of the yellow time that is regarded as red
            'offset' : float
                The program offset (used for coordinating several programs)

            These parameters are only used for traffic lights that are
            controlled by Aimsun (have 'Events enabled').
    """
    def __init__(self,
                 node_id,
                 turns,
                 signal_groups,
                 phases,
                 aimsun_controller_params={
                     'yellow_green_to_red': 3.,
                     'yellow_red_to_green': 0.,
                     'red_percentage': 50,
                     'offset': 0
                 }):
        self.node_id = node_id
        self.turns = turns
        self.signal_groups = signal_groups
        self.phases = phases
        self.aimsun_controller_params = aimsun_controller_params

        # Mapping from states to phases
        self.state_to_phase = {phase.colors: phase for phase in phases}

    def find_phase_by_state(self, state):
        """ Finds the traffic light phase object given the state """
        return self.state_to_phase[state]

    def update_aimsun_controller_params(self, **kwargs):
        """ Update the default Aimsun controller params. """
        self.aimsun_controller_params.update(**kwargs)

    def get_completed_phase_list(self):
        pass #TODO


# ==============================================================================================
# ================================= PhaseType enum  ============================================

MAJOR_PHASE_THRESHOLD = 5

class PhaseType(IntEnum):
    GREEN = 0
    IN_BETWEEN = 1

    @staticmethod
    def get_phases_type(phases: list):
        """
        Identify major (green) / minor type for all given phases.

        Args:
            phases (list): list of phases, phase can be a dict or wolf.Phase.

        Raises:
            TypeError: unrecognized phase type.

        Returns:
            list[wolf.PhaseType]: list of phase types
        """
        phases_type = []
        phases_colors = []
        if isinstance(phases[0], dict):
            phases_colors = [phase["colors"] for phase in phases]
        elif isinstance(phases[0], Phase):
            phases_colors = [phase.colors for phase in phases]
        else:
            raise TypeError("Argument should be a list of dict <colors, duration> or a list of wolf.Phase.")

        # if one movement always be green, ignore it when determining the major/minor phase
        ignored_movements = list(range(len(phases_colors[0])))
        for phase_colors in phases_colors:
            if len(ignored_movements) == 0:
                break
            for i, idx in enumerate(ignored_movements):
                if phase_colors[idx] in 'yr':
                    ignored_movements.pop(i)

        # determine major / minor phase
        if isinstance(phases[0], dict):
            for phase in phases:
                phase_colors = list(copy.deepcopy(phase["colors"]).lower())
                for i in ignored_movements[::-1]:
                    phase_colors.pop(i)

                if 'g' in phase_colors and phase["duration"] > MAJOR_PHASE_THRESHOLD:
                    # major phase
                    phases_type.append(PhaseType.GREEN)
                else:
                    phases_type.append(PhaseType.IN_BETWEEN)
        else:
            for phase in phases:
                phase_colors = list(copy.deepcopy(phase.colors).lower())
                for i in ignored_movements[::-1]:
                    phase_colors.pop(i)

                if phase.min_time == phase.max_time and phase.min_time <= MAJOR_PHASE_THRESHOLD:
                    phases_type.append(PhaseType.IN_BETWEEN)
                else:
                    phases_type.append(PhaseType.GREEN)

        return phases_type



# ==============================================================================================
# ==================================  Other utils  =============================================

def get_SUMO_links_to_turns_mapping_from_XML(path, tl_id__):
    # Temp., to be deleted
    net = etree.parse(path).getroot()
    phases = {}
    link_idx_to_turn = {}
    turn_to_link_idx = {}
    for tl_logic in net.findall('tlLogic'):
        tl_id = tl_logic.get('id')
        phases[tl_id] = []
        for phase in tl_logic.findall('phase'):
            phases[tl_id].append(phase.get('state'))
        link_idx_to_turn[tl_id] = {}
        turn_to_link_idx[tl_id] = {}

    for connection in net.findall('connection'):
        tl_id = connection.get('tl')
        if tl_id is None: continue
        link_idx = int(connection.get('linkIndex'))
        via_lane = connection.get('via')
        turn = via_lane[:via_lane.rindex('_')]
        link_idx_to_turn[tl_id][link_idx] = turn
        turn_to_link_idx[tl_id].setdefault(turn, []).append(link_idx)

    print(tl_id__)
    for phase in phases[tl_id__]:
        for turn, link_indices in turn_to_link_idx[tl_id__].items():
            if not all(phase[idx] == phase[link_indices[0]] for idx in link_indices):
                print(','.join(phase[idx] for idx in link_indices))




#TODO: Make 'common template' phasing schemes:
# * Four-way
#   Protected lefts
#   Permitted lefts
#   Split phasing (all movements from a single approach, alternating)
# * T-junction

#TODO: Make 'non-conflicting movements' matrix
# (and also divide the movements into rings and barriers?)

# TODO
def map_state_seq_to_aimsun_phase_seq(states): pass
def map_aimsun_phase_seq_to_state_seq(phases): pass
# TODO: Nema signal ring representation of phases
