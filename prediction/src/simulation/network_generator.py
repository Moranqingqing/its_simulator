import sumolib
import os
import subprocess
import tempfile

class Node:

    def __init__(self, nid, x, y, nodeType):
        self.nid = nid
        self.x = x
        self.y = y
        self.nodeType = nodeType

class Lane:

    def __init__(self, dirs=None, allowed=None, disallowed=None):
        self.dirs = dirs
        self.allowed = allowed
        self.disallowed = disallowed
        if self.dirs is None:
            self.dirs = []

class Edge:

    def __init__(self, eid=None, fromNode=None, toNode=None, numLanes=None, maxSpeed=None, lanes=None, shape=None,
                 spreadType=None):
        self.eid = eid
        self.fromNode = fromNode
        self.toNode = toNode
        self.numLanes = numLanes
        self.maxSpeed = maxSpeed
        self.lanes = lanes
        if self.lanes is None:
            self.lanes = [Lane() for _ in range(numLanes)]
        if numLanes is None:
            self.numLanes = len(self.lanes)
        self.shape = shape
        self.spreadType = spreadType

class Net:

    def __init__(self, lat_lon=True):
        self._nodes = {}
        self._edges = {}
        self.netName = None
        self.lat_lon = lat_lon

    def addNode(self, n):
        self._nodes[n.nid] = n

    def addEdge(self, e):
        self._edges[e.eid] = e

    def build(self, netName):

        nodesFile = tempfile.NamedTemporaryFile(mode="w", delete=False)
        print("<nodes>", file=nodesFile)
        for nid in self._nodes:
            n = self._nodes[nid]
            print('    <node id="%s" x="%s" y="%s" nodeType="%s"/>' % (
                n.nid, n.x, n.y, n.nodeType), file=nodesFile)
        print("</nodes>", file=nodesFile)
        nodesFile.close()

        edgesFile = tempfile.NamedTemporaryFile(mode="w", delete=False)
        print("<edges>", file=edgesFile)
        for eid in self._edges:
            e = self._edges[eid]
            print('    <edge id="%s" from="%s" to="%s" numLanes="%s" speed="%s" shape="%s" spreadType="%s">' % (
                e.eid, e.fromNode.nid, e.toNode.nid, e.numLanes, e.maxSpeed, e.shape, e.spreadType), file=edgesFile)
            print('    </edge>', file=edgesFile)

            hadConstraints = False
            for i, l in enumerate(e.lanes):
                if l.allowed is None and l.disallowed is None:
                    continue
                hadConstraints = True

        print("</edges>", file=edgesFile)
        edgesFile.close()

        netconvert = sumolib.checkBinary("netconvert")
        options = [netconvert, "-v", "-n", nodesFile.name, "-e", edgesFile.name, "-o", netName]

        if self.lat_lon:
            options += ["--proj.utm", "true"]
            # options += ["--proj", "+proj=utm +zone=17 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"]

        subprocess.call(options)
        os.remove(nodesFile.name)
        os.remove(edgesFile.name)
        self.netName = netName
