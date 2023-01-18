import sys
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

import randomTrips
from sumolib.net.generator.demand import *
from src.simulation.utils import *

randomTripsPath = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
sys.path.append(randomTripsPath)

RANDOMSEED = "42"
ROUTE_SUFFIX = ".rou.xml"
TRIPS_SUFFIX = ".trip.rou.xml"

DUAROUTER = sumolib.checkBinary("duarouter")

try:
    basestring
    # Allows isinstance(foo, basestring) to work in Python 3
except NameError:
    basestring = str

class DemandProfile(ABC):
    def __init__(self, net, b, e):

        self.net = net

        self.path = net.path
        self.net_file = net.net_file
        self.b = b
        self.e = e

    @abstractmethod
    def run(self):
        pass

    def log(self):
        with open(self.path + "logs.txt", "w") as f:
            for k, v in self.__dict__.items():
                f.write('"%s": "%s"\n' % (k, v))

class NetDemand(DemandProfile):
    """
    :param path:
    :param net:
    :param kwargs: flow_model, additional model arguments.
    """

    def __init__(self, net, b, e, flow_model, **kwargs):
        super().__init__(net, b, e)

        self.trip_file = self.path + flow_model + TRIPS_SUFFIX
        self.route_file = self.path + flow_model + ROUTE_SUFFIX
        self.flow_model = flow_model

        self.entrance, self.exit = get_gates(net)
        self.demand = Demand()

        if flow_model == "static":
            stream = Stream(sid=None, validFrom=None, validUntil=None, numberModel=kwargs["numberModel"],
                            departEdgeModel=self.entrance, arrivalEdgeModel=self.exit, vTypeModel="default",
                            via=None)
            self.demand.addStream(stream)

        elif flow_model == "wave":
            model = WaveComposition(offset=kwargs['offset'], curves=kwargs['curves'])
            stream = Stream(sid=None, validFrom=None, validUntil=None, numberModel=model,
                            departEdgeModel=self.entrance, arrivalEdgeModel=self.exit, vTypeModel="default",
                            via=None)
            self.demand.addStream(stream)

        elif flow_model == "linear":
            time = kwargs['time']
            model = kwargs['numberModel']
            model2 = LinearChange(model[0], model[1], time[1], time[2])

            stream1 = Stream(sid=None, validFrom=time[0], validUntil=time[1], numberModel=model[0],
                             departEdgeModel=self.entrance, arrivalEdgeModel=self.exit, vTypeModel="default",
                             via=None)

            stream2 = Stream(sid=None, validFrom=time[1], validUntil=time[2], numberModel=model2,
                             departEdgeModel=self.entrance, arrivalEdgeModel=self.exit, vTypeModel="default",
                             via=None)

            stream3 = Stream(sid=None, validFrom=time[2], validUntil=time[3], numberModel=model[1],
                             departEdgeModel=self.entrance, arrivalEdgeModel=self.exit, vTypeModel="default",
                             via=None)
            self.demand.addStream(stream1)
            self.demand.addStream(stream2)
            self.demand.addStream(stream3)

        vehicles = []
        for s in self.demand.streams:
            vehicles.extend(s.toVehicles(int(self.b), int(self.e), len(vehicles), None))

        with open(self.trip_file, 'w') as f:

            sumolib.writeXMLHeader(f, "$Id$", "routes")
            f.write('    <vType id="%s" vClass="%s"%s/>\n' % ("default", "passenger", ""))

            for v in sorted(vehicles, key=lambda veh: veh.depart):
                f.write('    <trip id="%s" depart="%s" from="%s" to="%s" type="%s" %s/>\n' %
                        (v.id, v.depart, v.fromEdge, v.toEdge, v.vType, ""))
            f.write("</routes>")

    def run(self, sampleFactor=None):

        subprocess.call([DUAROUTER, "-v",
                         "-n", self.net_file,
                         "--route-files", self.trip_file,
                         "-o", self.route_file,
                         "-b", self.b,
                         "-e", self.e])

    def get_trip_file(self):

        vehicles = []
        for s in self.demand.streams:
            vehicles.extend(s.toVehicles(int(self.b), int(self.e), len(vehicles), None))

        with open(self.trip_file, 'w') as f:

            sumolib.writeXMLHeader(f, "$Id$", "routes")
            f.write('    <vType id="%s" vClass="%s"%s/>\n' % ("default", "passenger", ""))

            for v in sorted(vehicles, key=lambda veh: veh.depart):
                f.write('    <trip id="%s" depart="%s" from="%s" to="%s" type="%s" %s/>\n' %
                        (v.id, v.depart, v.fromEdge, v.toEdge, v.vType, ""))
            f.write("</routes>")

class RandomDemand(DemandProfile):

    def __init__(self, net, b, e):
        super().__init__(net, b, e)
        self.demand_type = "random"

    def run(self, name="random", **kwargs):

        self.arguments = kwargs

        randomTripsCalls = []
        # have to have -n for netfile. --net-file input will not work
        options = ["-n", self.net_file,
                   "-r", self.path + name + ROUTE_SUFFIX,
                   "-t", "departLane=\"best\" departSpeed=\"max\" departPos=\"random\"",
                   "-b", self.b, "-e", self.e,
                   "-s", RANDOMSEED]

        for k, v in kwargs.items():
            if k == "period":
                options += ["--period", v]
            if k == "fringeFactor":
                options += ["--fringe-factor", v]
            elif k == "lengthFactor":
                options += ["-l", v]
            elif k == "speedFactor":
                options += ["--speed-exponent", v]

        random_options = randomTrips.get_options(options)
        randomTrips.main(random_options)
        randomTripsCalls.append(options)
        # create a batch file for reproducing calls to randomTrips.py
        SUMO_HOME_VAR = "%SUMO_HOME%"

        randomTripsPath = os.path.join(
            SUMO_HOME_VAR, "tools", "randomTrips.py")

        batchFile = self.path + "build.bat"
        with open(batchFile, 'w') as f:
            for opts in sorted(randomTripsCalls):
                f.write("python \"%s\" %s\n" %
                        (randomTripsPath, " ".join(map(quoted_str, getRelative(opts, self.path)))))

class ActivitygenDemand(DemandProfile):

    def __init__(self, net, b, e, template_file, stat_file='activitygen.stat.xml', pop=10000):

        """
        :param net:
        :param template_file: template to build activitygen configuration
                              from e.g. "data/Sumo/templates/additional_files-template.add.xml"
        :param pop: city population
        """

        super().__init__(net, b, e)

        self.stat_file = self.path + stat_file
        self.template_file = template_file

        # Demographics of the population
        children = int(0.25 * pop)
        working_adults = int(0.6 * pop)
        over65 = int(pop - children - working_adults)

        self.general = {"inhabitants": "1000",
                        "households": "500",
                        "childrenAgeLimit": "18",
                        "retirementAgeLimit": "65",
                        "carRate": "0.65",
                        "unemploymentRate": "0.05",
                        "footDistanceLimit": "500",
                        "incomingTraffic": "200",
                        "outgoingTraffic": "50"}

        self.parameters = {"carPreference": "0.75",
                           "meanTimePerKmInCity": "360",
                           "freeTimeActivityRate": "0.15",
                           "uniformRandomTraffic": "0.20",
                           "departureVariation": "120"}

        self.pop1 = {"beginAge": "0", "endAge": "18", "peopleNbr": str(children)}
        self.pop2 = {"beginAge": "19", "endAge": "64", "peopleNbr": str(working_adults)}
        self.pop3 = {"beginAge": "65", "endAge": "95", "peopleNbr": str(over65)}

        self.opening1 = {"hour": "0", "proportion": "0.30"}
        self.opening2 = {"hour": "3600", "proportion": "0.70"}
        self.closing1 = {"hour": "28800", "proportion": "0.20"}
        self.closing2 = {"hour": "30000", "proportion": "0.20"}
        self.closing3 = {"hour": "35000", "proportion": "0.60"}

        # Specify the density of people and work in each street of the city
        self.street = {}
        for i, edge in enumerate(net.sumo_net.getEdges()):
            self.street["street" + str(i)] = {"edge": edge.getID(), "population": "2.5", "workPosition": "10.0"}

        # Define the entry and exit edges of the network
        # These will be the city gates
        self.entrance = []
        for node, in_degree in dict(net.G.in_degree).items():
            if in_degree == 0:
                node1, node2, attr = list(net.G.out_edges(node, data=True))[0]
                self.entrance.append(attr['linkID'])

        self.exit = []
        for node, out_degree in dict(net.G.out_degree).items():
            if out_degree == 0:
                node1, node2, attr = list(net.G.in_edges(node, data=True))[0]
                self.exit.append(attr['linkID'])

        self.entry = {}
        self.out = {}
        for i, edge in enumerate(self.entrance):
            self.entry["entry" + str(i)] = {"edge": edge, "pos": "0.00", "incoming": "0.5", "outgoing": "0"}
        for i, edge in enumerate(self.exit):
            self.entry["entry" + str(i)] = {"edge": edge, "pos": "0.00", "incoming": "0.5", "outgoing": "0"}

    def build_tree(self):

        tree = ET.parse(self.template_file)
        root = tree.getroot()

        ET.SubElement(root, "general", self.general)
        ET.SubElement(root, "parameters", self.parameters)
        population = ET.SubElement(root, "population")
        ET.SubElement(population, "bracket", self.pop1)
        ET.SubElement(population, "bracket", self.pop2)
        ET.SubElement(population, "bracket", self.pop3)

        workHours = ET.SubElement(root, "workHours")
        ET.SubElement(workHours, "opening", self.opening1)
        ET.SubElement(workHours, "opening", self.opening2)
        ET.SubElement(workHours, "closing", self.closing1)
        ET.SubElement(workHours, "closing", self.closing2)
        ET.SubElement(workHours, "closing", self.closing3)

        streets = ET.SubElement(root, "streets")
        for i, edge in enumerate(self.net.sumo_net.getEdges()):
            ET.SubElement(streets, "street", self.street["street" + str(i)])

        cityGates = ET.SubElement(root, "cityGates")
        for i, edge in enumerate(self.entrance):
            ET.SubElement(cityGates, "entrance", self.entry["entry" + str(i)])
        for i, edge in enumerate(self.exit):
            ET.SubElement(cityGates, "entrance", self.out["entry" + str(i)])

        tree.write(self.stat_file)

    def run(self, output='activitygen'):

        output_file = self.path + output + TRIPS_SUFFIX

        ACTIVITYGEN = sumolib.checkBinary("activitygen")
        subprocess.call([ACTIVITYGEN, "-v",
                         "--net-file", self.net_file,
                         "--stat-file", self.stat_file,
                         "-o", output_file,
                         "--begin", self.b, "--end", self.e])
