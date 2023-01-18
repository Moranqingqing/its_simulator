from enum import Enum

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class DetectorActivation:
    POSITION = 'position'
    ENTRY = 'entry'
    EXIT = 'exit'


class MavType(Enum):
    """ Moving average types """
    NONE = 0
    EXPONENTIAL = 1
    SIMPLE = 2

class StatType(Enum):
    """ Statistic types """
    SPEED = 0
    FLOW = 1
    DENSITY = 2
