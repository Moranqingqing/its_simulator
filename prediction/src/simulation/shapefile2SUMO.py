from src.simulation.network_generator import *
import geopandas as gpd
import networkx as nx
import pyproj

class Shapefile2Sumo:

    def __init__(self, netName, shapefile):

        self.G = nx.read_shp(shapefile)
        self.df = gpd.read_file(shapefile)
        self.netName = netName

class AimsunShapefile2Sumo(Shapefile2Sumo):

    def __init__(self, netName="data/qew/qew.net.xml", shapefile='data/qew/shapefiles/gksection.shp', lat_lon=True):
        super().__init__(netName, shapefile)

        self.net = Net(lat_lon)

        for fnode, tnode, data in self.G.edges.data():

            lon1, lat1 = fnode
            node1 = Node(data['fnode'], lon1, lat1, "priority")
            self.net.addNode(node1)

            lon2, lat2 = tnode
            node2 = Node(data['tnode'], lon2, lat2, "priority")
            self.net.addNode(node2)

            coords = self.get_coords(data['linkID'])
            shape = get_shape(coords)
            edge = Edge(eid=data['linkID'], fromNode=node1, toNode=node2, numLanes=int(data['nb_lanes']),
                        maxSpeed=data['speed'], lanes=None, shape=shape, spreadType="center")
            self.net.addEdge(edge)

        self.net.build(netName=netName)

    def get_coords(self, eid):
        return list(self.df.loc[self.df['linkID'] == eid, 'geometry'].item().coords)

class HEREshapefile2sumo(Shapefile2Sumo):
    def __init__(self, shapefile='data/HERE/here_partial.shp', netName='data/Sumo/HERE/here2.net.xml'):
        super().__init__(netName, shapefile)

        self.net = Net()

        # define node ids
        nid = {}
        for i, node in enumerate(self.G.nodes):
            nid[node] = i
            self.G.nodes[node]['nid'] = i

        for fnode, tnode, data in self.G.edges.data():

            lon1, lat1 = fnode
            node1 = Node(nid[fnode], lon1, lat1, "priority")
            self.net.addNode(node1)

            lon2, lat2 = tnode
            node2 = Node(nid[tnode], lon2, lat2, "priority")
            self.net.addNode(node2)

            # Get linestring coordinates and convert into the proper shape format
            coordsF = self.get_coords(data['LINK_ID'])
            shapeF = get_shape(coordsF)

            # Reverse the coordinates and find the shape for the opposite direction
            coordsT = list(reversed(coordsF))
            shapeT = get_shape(coordsT)

            if data['DIR_TRAVEL'] == 'B':

                # Get edge IDs
                eidT = str(data['LINK_ID']) + 'T'
                eidF = str(data['LINK_ID']) + 'F'

                # Define edges for both directions
                edgeF = Edge(eid=eidF, fromNode=node1, toNode=node2, numLanes=data['FROM_LANES'],
                             maxSpeed=data['SPD_LIM'] / 3.6, lanes=None, shape=shapeF, spreadType="roadCenter")

                edgeT = Edge(eid=eidT, fromNode=node2, toNode=node1, numLanes=data['TO_LANES'],
                             maxSpeed=data['SPD_LIM'] / 3.6, lanes=None, shape=shapeT, spreadType="roadCenter")

                self.net.addEdge(edgeF)
                self.net.addEdge(edgeT)

            if data['DIR_TRAVEL'] == 'T':

                eid = str(data['LINK_ID']) + 'T'

                edge = Edge(eid=eid, fromNode=node2, toNode=node1, numLanes=data['TO_LANES'],
                            maxSpeed=data['SPD_LIM'] / 3.6, lanes=None, shape=shapeT, spreadType="center")
                self.net.addEdge(edge)

            elif data['DIR_TRAVEL'] == 'F':
                eid = str(data['LINK_ID']) + 'F'

                edge = Edge(eid=eid, fromNode=node1, toNode=node2, numLanes=data['FROM_LANES'],
                            maxSpeed=data['SPD_LIM'] / 3.6, lanes=None, shape=shapeF, spreadType="center")
                self.net.addEdge(edge)

        self.net.build(netName)

    def get_coords(self, eid):
        return list(self.df.loc[self.df['LINK_ID'] == eid, 'geometry'].item().coords)


def get_shape(coords, lat_lon=True):

    shape = ''
    for c in coords:
        lon, lat = c

        if lat_lon:
            x = lon
            y = lat
        else:
            x, y = convertlatlon2xy(lon, lat)

        shape += str(x) + ',' + str(y) + ' '

    return shape

def convertlatlon2xy(lon, lat):

    projParameter = "+proj=utm +zone=17 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    x_off, y_off = [-621667.40, -4839432.85]
    x, y = pyproj.Proj(projparams=projParameter)(lon, lat)

    return x + x_off, y + y_off


