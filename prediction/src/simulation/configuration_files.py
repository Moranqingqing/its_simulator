from abc import ABC
import subprocess
import sumolib
from src.simulation.utils import quoted_str, getRelative

TRIP_SUFFIX = ".trip.rou.xml"
ROUTE_SUFFIX = ".rou.xml"
NET_SUFFIX = ".net.xml"
ADD_SUFFIX = ".add.xml"
DUAROUTER_SUFFIX = ".duarcfg"
SUMO_SUFFIX = ".sumocfg"

ConfigPath = r"C:\Users\natal\Documents\Grad studies\MEng " \
             r"Project\Traffic-Congestion-Data-Mining\src\simulation\configuration_files.py"


class Config(ABC):

    def __init__(self, net, additional_files, begin, end, **kwargs):
        """
        :param net: net is an instance of the SUMONetwork object for the simulation being configured
        """

        self.path = net.path
        self.net_file = net.net_file
        self.additional_files = additional_files

        # Define a naming convention for files
        self.filename = self.net_file[:self.net_file.find('.')]

        self.config_file = None
        self._args = None

        for arg in kwargs.values():
            if type(arg) == list:
                self._args.append(arg)

        self.begin = begin
        self.end = end

    def run(self):
        subprocess.call(self._args)

    def get_batch(self, filename):
        batchFile = self.path + filename
        with open(batchFile, 'w') as f:
            f.write("python \"%s\" %s\n" %
                    (ConfigPath, " ".join(map(quoted_str, getRelative(self._args, self.path)))))


class DuarouterConfig(Config):

    def __init__(self, net, trip_files, output_file, additional_files, begin, end, **kwargs):
        super().__init__(net, additional_files, begin, end, **kwargs)
        self.config_file = self.filename + DUAROUTER_SUFFIX

        self.trip_files = trip_files
        self.route_files = output_file + ROUTE_SUFFIX

        self._duarouter = sumolib.checkBinary("duarouter")
        self._args = [self._duarouter,
                      "-C", self.config_file,
                      "--net-file", self.net_file,
                      "--additional-files", self.additional_files,
                      "--route-files", self.trip_files,
                      "-o", self.route_files,
                      "--repair", "true",
                      "--departlane", "best",
                      "--begin", self.begin, "--end", self.end]

class SimulationConfig(Config):
    def __init__(self, net, route_files, additional_files, begin, end, scale="1", **kwargs):

        """
        :param net: SUMONetwork instance
        :param route_files: .rou.xml file
        :param additional_files: .add.xml file
        :param begin: begin time of simulation in seconds
        :param end: end time of simulation in seconds
        :param scale: demand scale
        :param kwargs: If a list is given, it is assumed to be a list of additional arguments given to the simulation
        """

        super().__init__(net, additional_files, begin, end, **kwargs)

        self.demand_type = route_files[:route_files.find('.')]
        self.config_file = self.path + self.filename + '_' + self.demand_type + SUMO_SUFFIX
        self.output_file = self.path + "simulation_output.xml"
        self.route_files = route_files
        self.scale = scale

        self._sumo = sumolib.checkBinary("sumo-gui")
        self._args = [self._sumo,
                      "-C", self.config_file,
                      "--net-file", self.net_file,
                      "--route-files", self.route_files,
                      "--additional-files", self.additional_files,
                      "--begin", self.begin, "--end", self.end,
                      "--scale", self.scale,
                      "--start", "false"]


