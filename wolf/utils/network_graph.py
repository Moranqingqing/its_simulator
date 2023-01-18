"""
*** WARNING: Work-in-progress, may have errors or incomplete methods ***

Builds a representation of a traffic network as a directed graph.
The representation is agnostic to SUMO or Aimsun.

To build for SUMO: use the build_network_graph_from_SUMO_XML function
To build for Aimsun: call the build_network_graph_on_aimsun_load function
                     from the load.py module in flow/utils/aimsun/
"""
import json
from lxml import etree
from lxml.builder import ElementMaker
from collections import deque
from copy import deepcopy
import math
import numpy as np

RAD_TO_DEG = 180 / math.pi

SUMO_TRAFFIC_LIGHT_TYPES = {'traffic_light',
                            'traffic_light_unregulated',
                            'traffic_light_right_on_red'}
SUMO_GREEN_SIGNALS = {'G', 'g', 's'}
DIR = {
    'STRAIGHT': 's',
    'TURN': 't',
    'LEFT': 'l',
    'RIGHT': 'r',
    'PARTIAL_LEFT': 'L',
    'PARTIAL_RIGHT': 'R',
    'INVALID': 'invalid'
}
# Threshold for considering a direction to be straight
# (angle measured away from the vertical axis, 0 being parallel
#  with the incoming direction)
STRAIGHT_DIR_THRESHOLD_ANGLE = math.pi / 3


def parse_shape(shape_string):
    """
    Gets the shape string in the format 'x0,y0 x1,y1 ...', and parses it
    into the list of lists [[x0, y0], [x1, y1], ...], where xi and yi are floats.

    Arguments
    ---------
        shape_string: String
            'x0,y0 x1,y1 ...'

    Returns
    -------
        List of pairs of floats
            [[x0, y0], [x1, y1], ...]
    """
    shape = []
    if shape_string is not None:
        for coord_pair in shape_string.split(' '):
            [x, y] = [ float(coord) for coord in coord_pair.split(',') ]
            shape.append([x, y])
    return shape

def shape_str(shape):
    """
    Inverse operation to 'parse_shape'

    Arguments
    ---------
        shape : List of pairs of floats
            [[x0, y0], [x1, y1], ...]

    Returns
    -------
        String
            'x0,y0 x1,y1 ...'
    """
    return ' '.join( f'{str(x)},{str(y)}' for x, y in shape)

def translate_shape(shape, dX, dY):
    """
    Uniformly translates every point of a shape by the fixed
    vector (dX, dY).

    Arguments
    ---------
        shape : List of pairs of floats
            [[x0, y0], [x1, y1], ...]
        dX, dY: floats
            The translation vector

    Returns
    -------
            [[x0 + dX, y0 + dY], [x1 + dX, y1 + dY], ...]
    """
    trans = lambda pt: (pt[0] + dX, pt[1] + dY)
    return map(trans, shape)

def measure_polyline_length(shape):
    """
    Finds the sum of the lengths of the line segments in a polyline
    """
    L = 0
    for [x0, y0], [x1, y1] in zip(shape, shape[1:]):
        L += (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0)
    return math.sqrt(L)


def find_vertex_centroid_and_radius(shape):
    """
    Finds the coordinates of the centroid of a polygon using the shoelace formula,
    then finds the radius of the smallest circle that is centered at the centroid
    and contains the polygon.

    Arguments
    ---------
        shape: list of lists of floats [[x0, y0], [x1, y1], ...]
            The corners of the polygon

    Returns
    -------
        x: float, y: float, R: float
            x, y coordinates of the centroid, and the radius of the intersection
    """
    if len(shape) == 0:
        return None, None, 0
    if len(shape) == 1:
        return shape[0][0], shape[0][1], 0
    if len(shape) == 2:
        x0, y0, x1, y1 = shape[0][0], shape[0][1], shape[1][0], shape[1][1]
        return (x0+x1)/2, (y0+y1)/2, math.sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))

    Cx = Cy = A = 0  # The centroid x- and y- coordinates, and area of the polygon
    shape.append(shape[0]) # Close the shape
    for (xi, yi), (xipp, yipp) in zip(shape, shape[1:]):
        cross = xi*yipp - xipp*yi
        Cx += (xi + xipp) * cross
        Cy += (yi + yipp) * cross
        A  += cross
    shape.pop() # Open the shape back up

    if A == 0:
        # Unexpected degenerate case when the points are on a line
        x0, y0, xN, yN = shape[0][0], shape[0][1], shape[-1][0], shape[-1][1]
        return (x0+xN)/2, (y0+yN)/2, 0

    # Coordinates of the centroid
    Cx /= 3*A
    Cy /= 3*A

    # Find the radius of the smallest enclosing circle centered at the centroid
    R = -math.inf
    for (xi, yi) in shape:
        R = max(R, (xi-Cx)*(xi-Cx) + (yi-Cy)*(yi-Cy))
    return Cx, Cy, math.sqrt(R)

def find_angle(x, y, round=True):
    """
    Finds the angle the line segment from (0, 0) to (x, y) makes
    with the positive y-direction. If round is True, the value is
    rounded to an integer.
    """
    rad = math.atan2(x, y) # (x, y) and not (y, x) because measuring from y-axis
    if rad < 0: rad += 2 * math.pi
    angle = rad * RAD_TO_DEG
    return (round(angle) if round else angle) % 360

def find_dir_vector(P, Q):
    """
    Find the direction vector along the vector from Q to P

    Arguments
    ---------
        P : np.array of shape (d,)
        Q : np.array of shape (d,)
            Two d-dimensional vectors
    Returns
    -------
        np.array
            The unit direction vector from Q to P
    """
    P, Q = np.array(P), np.array(Q)
    return (P - Q) / np.linalg.norm(P - Q)



def build_network_graph_from_SUMO_XML(XML_net):
    V = {}
    E = {}
    lanes_by_id = {}
    TLs = set()

    root = XML_net.getroot()

    # SUMO junctions correspond to the vertices of the Network Graph
    for junction in root.findall('junction'):
        vertex_id = junction.get('id')
        vertex_type = junction.get('type')
        if vertex_type in SUMO_TRAFFIC_LIGHT_TYPES:
            vertex_type = 'tl'
            TLs.add(vertex_id)

        polygon = parse_shape(junction.get('shape'))
        x, y, R = find_vertex_centroid_and_radius(polygon)

        V[vertex_id] = {'type': vertex_type,
                        'x': x,
                        'y': y,
                        'polygon': polygon,
                        'R': R,
                        'turns': {}}

    # Internal SUMO edges correspond to turning motions within a
    # vertex
    # External SUMO edges correspond to edges of the Network Graph
    #TODO: Turns should be the internal edges ... so what are the links?
    for edge in root.findall('edge'):
       edge_id = edge.get('id')

       not_a_vertex = edge.get('function') != 'internal'
       if not_a_vertex:
           edge_ = {'lanes': {}}

       lanes = edge.findall('lane')
       for lane in lanes:
           lane_id = lane.get('id')
           lane_shape = parse_shape(lane.get('shape'))
           lane_shape_len = measure_polyline_length(lane_shape)
           lane_ = {'shape': lane_shape,
                    'len': float(lane.get('length')),
                     # Logical length (len) may be different from geometric length (shape_len)
                    'shape_len': lane_shape_len,
                    'width': 3,
                    'speed_limit': float(lane.get('speed')),
                    'edge': edge_id}
           lanes_by_id[lane_id] = lane_
           if not_a_vertex: edge_['lanes'][lane_id] = lane_

       if not_a_vertex:
           edge_['from'] = edge.get('from')
           edge_['to'] = edge.get('to')
           edge_['middle_lane_shape'] = parse_shape(lanes[len(lanes) // 2].get('shape'))
           edge_['middle_lane_length'] = float(lanes[len(lanes) // 2].get('length'))
           edge_['thickness'] = 3 * len(lanes)
           E[edge_id] = edge_


    # Connections go between incoming and outgoing lanes
    internal_edge_connection_via = {}
    for connection in root.findall('connection'):
        via_lane_id = connection.get('via')
        if via_lane_id is None:
            continue

        from_edge_id = connection.get('from')
        from_lane_idx = connection.get('fromLane')

        if from_edge_id[0] == ':':
            # These connections are from an internal edge: record part of the data for the next step
            from_lane_id = from_edge_id + '_' + from_lane_idx
            internal_edge_connection_via[from_lane_id] = via_lane_id
        else:
            # These connections are from an external edge: Record in the graph

            vertex_id = via_lane_id[1: via_lane_id.rindex('_', 0, via_lane_id.rindex('_'))]
            # Internal lane id has the format ':vertexId_edgeIdx_laneIdx'

            link_idx = connection.get('linkIndex')
            if link_idx is None:
                link_idx = len(V[vertex_id]['turns'])
            link_idx = int(link_idx)

            V[vertex_id]['turns'][link_idx] = {'shape': lanes_by_id[via_lane_id]['shape'],
                                               'len':   lanes_by_id[via_lane_id]['len'],
                                               'speed_limit': lanes_by_id[via_lane_id]['speed_limit'],
                                               'lane_ids': [via_lane_id],
                                               'from': connection.get('from'),
                                               'from_lane_idx': int(connection.get('fromLane')),
                                               'to': connection.get('to'),
                                               'to_lane_idx': int(connection.get('toLane')),
                                               'dir': connection.get('dir'),
                                               'via': via_lane_id}

    # Several lanes may be involved in a single turn
    for vertex in V.values():
        for turn in vertex['turns'].values():
            via = turn['via']
            while via in internal_edge_connection_via:
                via = internal_edge_connection_via[via]
                turn['shape'] += lanes_by_id[via]['shape']
                turn['len']   += lanes_by_id[via]['len']
                turn['lane_ids'].append(via)
            del turn['via']  # No longer needed

    return NetworkGraph(V, E, lanes_by_id, TLs)


def build_network_graph_on_aimsun_load(nodes, sections, turnings, detectors, bbox):
    """
    Convention
    ----------
        Although the object ids in Aimsun are integers, the object ids
        are stored as strings, because:
            * Object ids in SUMO are strings
            * The keys are converted to strings during JSON encoding.
        The object ids have to be converted to integers for Aimsun API calls.
    """
    V = {}
    E = {}
    lane_id_to_lane_mapping = {}
    detr_dict = {}
    TLs = set()

    num_sinks = 0
    num_sources = 0

    # Aimsun nodes correspond to the vertices of the graph
    for n in nodes:
        vertex_id = str(n.getId())
        vertex_type = 'Aimsun' # TODO: Check if junction is controlled

        polygon = [ [pt.x, pt.y] for pt in n.getPolygon() ]
        x, y, R = find_vertex_centroid_and_radius(polygon)

        V[vertex_id] = {'type': vertex_type,
                        'x': x,
                        'y': y,
                        'polygon': polygon,
                        'R': R,
                        'turns': {}}

    # Aimsun sections correspond to the edges of the graph
    for s in sections:
        edge_id = str(s.getId())
        edge_speed_limit = s.getSpeed()
        edge_lanes = []
        edge_thickness = 0

        for lane_idx, lane in enumerate(s.getLanes()):
            shape = [ [pt.x, pt.y] for pt in s.calculateLanePolyline(lane_idx, True)]
            lane_id = f'{edge_id}_{lane_idx}'
            lane_ = {'id': lane_id,
                     'shape': shape,
                     'len': s.getLaneLength2D(lane_idx),
                     'width': s.getLaneWidth(lane_idx),
                     'speed_limit': edge_speed_limit,
                     'edge': edge_id}

            edge_thickness += lane_['width']

            lane_id_to_lane_mapping[lane_id] = lane_
            edge_lanes.append(lane_)

        mid_lane_idx = len(edge_lanes) // 2
        if len(edge_lanes) % 2 == 1:
            # If the edge has an odd number of lanes, record the shape and length
            # of the middle lane
            mid_lane_shape = edge_lanes[mid_lane_idx]['shape']
        else:
            # If the edge has an even number of lanes, record the average of
            # the coordinates of the shape of the middle two lanes
            mid_lane_shape = [
                [(x0+x1)/2, (y0+y1)/2] for (x0, y0), (x1, y1) in
                zip(edge_lanes[mid_lane_idx-1]['shape'], edge_lanes[mid_lane_idx]['shape'])
            ]
        mid_lane_length = s.length2D()

        edge_origin = s.getOrigin()
        if edge_origin is not None:
            edge_from = str(edge_origin.getId())
        else:
            # If the edge origin is None, add a corresponding source vertex
            source_vertex_id = f'source_{num_sources}'
            V[source_vertex_id] = {'type': 'dead_end',
                                   'x': mid_lane_shape[0][0],
                                   'y': mid_lane_shape[0][1],
                                   'polygon': [mid_lane_shape[0]],
                                   'R': 0.,
                                   'turns': {}}
            edge_from = source_vertex_id
            num_sources += 1

        edge_dest = s.getDestination()
        if edge_dest is not None:
            edge_to = str(edge_dest.getId())
        else:
            # If the edge destination is None, add a corresponding sink vertex
            sink_vertex_id = f'sink_{num_sinks}'
            V[sink_vertex_id] = {'type': 'dead_end',
                                 'x': mid_lane_shape[-1][0],
                                 'y': mid_lane_shape[-1][1],
                                 'polygon': [mid_lane_shape[-1]],
                                 'R': 0.,
                                 'turns': {}}
            edge_to = sink_vertex_id
            num_sinks += 1

        E[edge_id] = {
            'from': edge_from,
            'to': edge_to,
            'lanes': edge_lanes,
            'thickness': edge_thickness,
            'middle_lane_shape': mid_lane_shape,
            'middle_lane_length': mid_lane_length,
            'turn': False,
        }
        E[edge_id]['turns_L_to_R'] = (
            [str(t.getId()) for t in edge_dest.getFromTurningsOrderedFromLeftToRight(s)]
            if edge_dest is not None else [])

    # Register the turns
    for t in turnings:
        turn_id = str(t.getId())
        turn_vertex = str(t.getNode().getId())
        turn_polygon = t.getPolygon()
        turn_polyline = t.calculatePolyline()

        turn_ = {
            'from': str(t.getOrigin().getId()),
            'to': str(t.getDestination().getId()),
            'from_lane_indices': list(range(t.getOriginFromLane(), t.getOriginToLane()+1)),
            'to_lane_indices': list(range(t.getDestinationFromLane(), t.getDestinationToLane()+1)),
            'polygon': [[pt.x, pt.y] for pt in turn_polygon],
            'shape': [[pt.x, pt.y] for pt in turn_polyline],
            'len': turn_polyline.length2D(),
            'speed_limit': t.getSpeed(),
            'turn': True,
        }
        E[turn_id] = turn_
        V[turn_vertex]['turns'][turn_id] = turn_
        lane_id_to_lane_mapping[turn_id] = turn_ #TODO: What to do here? -> Turns should be edges
                                                 # Use turning entities

    # Register the detectors
    for d in detectors:
        detr_id = str(d.getId())
        detr_pos = d.absolutePosition()
        detr_dict[detr_id] = {
            'edge_id': str(d.getSection().getId()),
            'lanes': list(range(d.getFromLane(), d.getToLane()+1)),
            'len': d.getLength(),
            'pos': d.getPosition(),
            'x': detr_pos.x,
            'y': detr_pos.y,
        }

    return NetworkGraph(
        V, E,
        detectors=detr_dict,
        lanes=lane_id_to_lane_mapping,
        bbox=bbox)


def build_network_graph_from_dict(dict):
    return NetworkGraph(
        dict['V'], dict['E'],
        lanes=dict['lanes'],
        detectors=dict['detectors'],
        bbox=dict['bbox'],
        classify_turn_dirs=True)

def build_network_graph_from_JSON(JSON_net):
    with open(JSON_net, 'r') as file:
        dct = json.load(file)
    return build_network_graph_from_dict(dct)


class NetworkGraph():
    """ Doc #TODO """
    def __init__(self,
                 V, E,
                 lanes=None,
                 detectors={},
                 bbox=None,
                 tl_ids=set(),
                 gen_lane_id_to_lane_mapping=False,
                 classify_turn_dirs=False):

        assert len(set(V.keys())) == len(V.keys()), \
            'The vertex set V passed to NetworkGraph contains duplicate vertex ids'
        assert len(set(E.keys())) == len(E.keys()), \
            'The edge set E passed to NetworkGraph contains duplicate edge ids'

        # Initialize the vertex set
        self.V = V
        for vertex in self.V.values():
            vertex['in_edges']  = []
            vertex['out_edges'] = []
            vertex['in_deg'] = 0
            vertex['out_deg'] = 0
            # Record the turning directions and destinations
            turning_dirs = {}
            turning_dests = {}
            for turn_id, turn in vertex['turns'].items():
                fr = turn['from']
                to = turn['to']

                turning_dirs.setdefault(fr, {}).setdefault(to, {}).setdefault('ids', []).append(turn_id)
                turning_dests.setdefault(to, {}).setdefault(fr, {}).setdefault('ids', []).append(turn_id)

            vertex['turning_directions'] = turning_dirs
            vertex['turning_destinations'] = turning_dests

        # Populate edges, and compute in- and out-degrees
        self.E = E
        for edge_id, edge in self.E.items():
            # Check that the 'from' and 'to' vertices are in the subnetwork #TODO: Add dead_ends?
            #TODO: Clarify the type of indices: Should all be strings?
            from_ = edge['from']
            to_   = edge['to']
            if from_ in self.V:
                self.V[from_]['out_edges'].append(edge_id)
            if to_ in self.V:
                self.V[to_]['in_edges'].append(edge_id)

        # Compute vertex degrees
        for vertex in self.V.values():
            vertex['out_deg'] = len(vertex['out_edges'])
            vertex['in_deg'] = len(vertex['in_edges'])

        # If required, classify turn directions (mainly useful for Aimsun)
        if classify_turn_dirs:
            for edge in self.E.values():
                if edge['turn']: continue
                turn_sweep = edge['turns_L_to_R']
                if len(turn_sweep) > 0:
                    # * The 'STRAIGHT' turning direction is the one whose end tip is closest
                    #   to being parallel with the end tip of the incoming edge, assuming that
                    #   there is such an angle that is below the STRAIGHT_DIR_THRESHOLD_ANGLE.
                    #   If there is no such angle, then there is no straight direction.
                    #   The rest of the directions are 'PARTIAL_LEFT' or 'PARTIAL_RIGHT',
                    #   according to where they are relative to the min. angle direction, except
                    # * The leftmost turn is 'LEFT' (unless it is STRAIGHT)
                    # * The rightmost turn is 'RIGHT' (unless it is STRAIGHT)

                    # Find the direction vector at the end of the edge shape
                    edge_dir = find_dir_vector(np.array(edge['middle_lane_shape'][-2]),
                                               np.array(edge['middle_lane_shape'][-1]))

                    # Can get away without this loop, but would be far less readable
                    turn_cosines = np.zeros(len(turn_sweep))
                    for idx, t in enumerate(turn_sweep):
                        turn = self.E[t]
                        turn_dir = find_dir_vector(np.array(turn['shape'][-2]),
                                                   np.array(turn['shape'][-1]))
                        turn_cosines[idx] = np.dot(edge_dir, turn_dir)
                    turn_angles = np.arccos(turn_cosines)
                    min_angle = np.min(turn_angles)
                    # from 0 to pi in either direction, 0 being parallel and pi anti-parallel
                    min_angle_idx = np.argwhere(turn_angles == min_angle)[0][0]

                    self.E[turn_sweep[0]]['dir'] = DIR['LEFT']
                    for turn_id in turn_sweep[1:min_angle_idx]:
                        self.E[turn_id]['dir'] = DIR['PARTIAL_LEFT']
                    for turn_id in turn_sweep[min_angle_idx+1:-2]:
                        self.E[turn_id]['dir'] = DIR['PARTIAL_RIGHT']
                    self.E[turn_sweep[-1]]['dir'] = DIR['RIGHT']

                    if min_angle < STRAIGHT_DIR_THRESHOLD_ANGLE:
                        self.E[turn_sweep[min_angle_idx]]['dir'] = DIR['STRAIGHT']

                    # STRAIGHT may overwrite one of LEFT or RIGHT. This can happen, for example,
                    # in a 3-way intersection with one straight and one left turn (or one straight
                    # and one right turn). An example with no straight directions is a 3-way
                    # end-of-the-road junction (with one left and one right turn).

        # Copy reference to (or create) the lane_id-to-lane mapping
        # TODO: Include turns in this in the Aimsun case (what should be the turn ids?)
        if lanes is not None:
            self.lanes = lanes
        elif gen_lane_id_to_lane_mapping:
            for edge_id, edge in self.E.items():
                for lane_idx, lane in enumerate(edge['lanes']):
                    lane_id = f'{edge_id}_{lane_idx}'
                    self.lanes[lane_id] = lane
        else:
            self.lanes = []

        # Find source and sink vertices
        self.source_ids = []
        self.sink_ids   = []
        for vertex_id, vertex in self.V.items():
            # FIXME: Not quite correct:
            if ((vertex['out_deg'] == 0 and vertex['in_deg'] > 0) or
                vertex['type'] == 'dead_end'):
                self.sink_ids.append(vertex_id)
            if ((vertex['in_deg'] == 0 and vertex['out_deg'] > 0) or
                vertex['type'] == 'dead_end'):
                self.source_ids.append(vertex_id)

        self.tl_ids = tl_ids
        self.detectors = detectors
        self.bbox = bbox
        # Done __init__

    def get_incident_edges(self, vertices):
        """
        Returns the ids of all in- and out- edges for the vertices
        in the passed list.

        Arguments
        ---------
            vertices : List of Strings
                List of ids of graph vertices

        Returns
        -------
            Set
                Set of ids of all edges incident to at least one of the vertices
                in the list
        """
        incident_edges = set()
        for v in vertices:
            v = self.V[v]
            incident_edges = incident_edges.union(set(v['in_edges'])).union(set(v['out_edges']))
        return incident_edges

    def get_source_vertex_ids(self):
        return self.source_ids

    def get_source_vertices(self):
        return [self.V[vertex_id] for vertex_id in self.source_ids]

    def get_sink_vertex_ids(self):
        return self.sink_ids

    def get_sink_vertices(self):
        return [self.V[vertex_id] for vertex_id in self.sink_ids]

    def get_entering_edge_ids(self):
        entering_edge_ids = []
        for vertex_id in self.source_ids:
            entering_edge_ids += self.V[vertex_id]['out_edges']
        return entering_edge_ids

    def get_entering_edges(self):
        return [self.E[edge_id] for edge_id in self.get_entering_edge_ids()]

    def get_entering_lane_ids(self):
        entering_lane_ids = []
        for edge in self.get_entering_edges():
            entering_lane_ids += edge['lanes'].keys()
        return entering_lane_ids

    def get_entering_lanes(self):
        return [self.lanes[lane_id] for lane_id in self.get_entering_lane_ids()]

    def get_exiting_edge_ids(self):
        exiting_edge_ids = []
        for vertex_id in self.sink_ids:
            exiting_edge_ids += self.V[vertex_id]['in_edges']
        return exiting_edge_ids

    def get_exiting_edges(self):
        return [self.E[edge_id] for edge_id in self.get_exiting_edge_ids()]

    def get_exiting_lane_ids(self):
        exiting_lane_ids = []
        for edge in self.get_exiting_edges():
            exiting_lane_ids += edge['lanes'].keys()
        return exiting_lane_ids

    def get_exiting_lanes(self):
        return [self.lanes[lane_id] for lane_id in self.get_exiting_lane_ids()]


    def get_induced_subgraph(self, vertices, gen_lane_id_to_lane_mapping=False):
        """
        Constructs the induced subgraph on a collection of vertices.

        Induced subgraph on a subset S of the vertex set is a graph that contains every
        edge of the Network Graph that connects two vertices both of which are in S.

        Arguments
        ---------
            vertices : Iterable of Strings
                The ids of the vertices in the vertex set
            gen_lane_id_to_lane_mapping : Boolean
                Whether to generate a lane-mapping for the induced graph

        Returns
        -------
            NetworkGraph
                The induced subgraph
        """
        vertex_copies = {}
        induced_edges = {}
        for v in vertices:
            vertex_copies[v] = deepcopy(self.V[v])
            for e in self.V[v]['in_edges']:
                if e not in induced_edges and self.E[e]['from'] in vertices:
                    induced_edges[e] = deepcopy(self.E[e])
            for e in self.V[v]['out_edges']:
                if e not in induced_edges and self.E[e]['to'] in vertices:
                    induced_edges[e] = deepcopy(self.E[e])

        return NetworkGraph(vertex_copies,
                            induced_edges,
                            gen_lane_id_to_lane_mapping=gen_lane_id_to_lane_mapping)

    def find_k_nbhd(self, vertices, k, induced_subgraph=True):
        """
        Finds all vertices within a path of length k of the passed collection
        of vertices (the k-neighbourhood of the vertices).

        Returns either the ids of vertices in the k-neighbourhood or the
        induced subgraph on the k-neighbourhood, depending on the 'induced_subgraph'
        argument.

        Arguments
        ---------
            vertices : Iterable of Strings
                The ids of the vertices to find the k-neighbourhood of
            k : Integer
                The number of hops to include in the neighbourhood
            induced_subgraph : Boolean
                If True, the induced subgraph on the k-nbhd is returned;
                if False, only the list of vertex ids in the k-neighbourhood

        Returns
        -------
            List of Strings or NetworkGraph
        """
        # Keep track of the shortest number of hops from one of the vertices
        # in the initial set to a subset of the remaining vertices.
        # -1 means not yet visited
        hops = dict.fromkeys(self.V.keys(), -1)
        nbhd = list(vertices)
        for v in vertices:
            hops[v] = 0

        def add_vertex_and_update_queue(w, H):
            if hops[w] == -1:
                # If the vertex was not added to the k-neighbourhood, add it
                nbhd.append(w)
                hops[w] = H
            if hops[w] > H:
                # If found a shorter path to the vertex, update the path length
                hops[w] = H

        idx = 0
        v = nbhd[idx]
        while hops[v] < k:
            # Vertices in a breadth-first traversal increase monotonically in depth
            # Stop when the first vertex with distance k is reached, as
            # no more vertices of distance <= k will be added
            h = hops[v]
            H = h + 1
            for e in self.V[v]['out_edges']:
                w = self.E[e]['to']
                add_vertex_and_update_queue(w, H)
            for e in self.V[v]['in_edges']:
                w = self.E[e]['from']
                add_vertex_and_update_queue(w, H)
            idx += 1
            v = nbhd[idx]

        return self.get_induced_subgraph(nbhd) if induced_subgraph else nbhd


    def get_routes_upstream_to_edge(self,
                                    edge_id,
                                    detection_range,
                                    only_straight=True):
        """ TODO """
        edge = self.E[edge_id]
        from_vertex = self.V[edge['from']]
        delta = detection_range - edge['middle_lane_length']

        if any((delta <= 0,
                from_vertex['type'] == 'tl',
                edge_id not in from_vertex['turning_destinations'])):
            # Terminate if either:
            #    * Detection range is reached
            #    * The next upstream vertex is controlled by a traffic light
            #    * The next upstream vertex does not have the edge as an outgoing
            #      edge in any of its turning directions
            upstream_routes = [{'ids': [edge_id],
                                'leftover_length': delta}]
        else:
            upstream_routes = []
            for approach_id in from_vertex['turning_destinations'][edge_id].keys():

                # Keep track of the connecting lane lengths as well
                turn_ids = from_vertex['turning_destinations'][edge_id][approach_id]['ids']
                turns = [self.E[turn_id] for turn_id in turn_ids]

                # Optionally, filter out the non-straight approaches
                if only_straight:
                    turns = [turn for turn in turns if turn['dir'] == DIR['STRAIGHT']]
                    if len(turns) == 0:
                        continue

                # A route is a sequence of (non-internal) edges, without turns
                delta -= max(turn['len'] for turn in turns)
                tails = self.get_routes_upstream_to_edge(approach_id,
                                                         delta,
                                                         only_straight)
                for tail in tails:
                    tail['ids'].append(edge_id)
                upstream_routes.extend(tails)

        return upstream_routes

    def get_turning_dirs_for_phase(self, tl_id, phase_state):
        turns = self.V[tl_id]['turns']
        turning_dirs_for_phase = set()
        for signal_idx, signal in enumerate(phase_state):
            if (signal in SUMO_GREEN_SIGNALS) and signal_idx in turns:
                turning_dirs_for_phase.add( (turns[signal_idx]['from_edge'],
                                             turns[signal_idx]['to_edge']) )
        return turning_dirs_for_phase

    def get_routes_upstream_to_phase(self,
                                     tl_id,
                                     phase_state,
                                     detection_range,
                                     only_straight=True):

        upstream_routes = []
        for incoming_edge, outgoing_edge in self.get_turning_dirs_for_phase(tl_id,
                                                                            phase_state):
            dir_incoming_routes = self.get_routes_upstream_to_edge(incoming_edge,
                                                                   detection_range,
                                                                   only_straight)
            for route in dir_incoming_routes:
                route['ids'].append(outgoing_edge)

            upstream_routes.extend(dir_incoming_routes)

        return upstream_routes


    def __get_upstream_edge_group_in_reverse(self,
                                             edge_id,
                                             detection_range):
        """
        Builds an edge group upstream of the passed edge, but with the edges
        ordered in reverse (with respect to the convention used in the Wolf kernel).
        This is to avoid adding elements to the start of a list (or using a deque).

        The get_upstream_edge_group method calls this one, then reverses the edge order,
        so should be used with the Wolf kernel. Cf. the docstring of get_upstream_edge_group.
        """
        edge = self.E[edge_id]
        assert not edge['turn']
        from_vertex = self.V[edge['from']]
        delta = detection_range - edge['middle_lane_length']

        straight_approaches = []
        more_than_one_straight_turn_in_approach = False
        turns_into_edge = from_vertex['turning_destinations'].get(edge_id) or {}

        for approach_id in turns_into_edge.keys():
            turn_ids = turns_into_edge[approach_id]['ids']
            # Filter out the non-straight directions
            turn_ids = [turn_id for turn_id in turn_ids if self.E[turn_id]['dir'] == DIR['STRAIGHT']]
            if len(turn_ids) > 0:
                if len(turn_ids) != 1:
                    more_than_one_straight_turn_in_approach = True
                    break
                straight_turn_id = turn_ids.pop()
                straight_approaches.append((approach_id, straight_turn_id))

        if any((delta <= 0,
                from_vertex['type'] == 'tl',
                len(straight_approaches) != 1,
                more_than_one_straight_turn_in_approach)):
            # Terminate if either:
            #    * The detection range is reached
            #    * The next upstream vertex is controlled by a traffic light
            #    * There is more than one straight approach
            #    * There is more than one straight turn in a straight approach
            return {'edge_group': [edge_id],
                    'leftover_length': delta}
        else:
            approach_id, turn_id = straight_approaches.pop()
            delta -= self.E[turn_id]['len']
            tail = self.__get_upstream_edge_group_in_reverse(approach_id, delta)
            tail['edge_group'].extend([turn_id, edge_id])
            return tail


    def get_upstream_edge_group(self,
                                edge_id,
                                detection_range):
        """ TODO """
        edge_group = self.__get_upstream_edge_group_in_reverse(edge_id, detection_range)
        edge_group['edge_group'].reverse()
        return edge_group


    def get_turning_lanes_for_edge_pair(self, fr, to):
        """
        Returns the set of ids of all lanes connecting edge 'fr' to edge 'to'
        """
        vertex = self.V[self.E[to]['from']]
        lanes = set()
        for key in vertex['turning_directions'][fr][to]['indices']:
            lanes |= set(vertex['turns'][key]['lane_ids'])
        return lanes

    def get_lengths_of_turning_lanes_for_edge_pair(self, fr, to):
        """
        Returns a dictionary
            key : id of a turning lane from edge 'fr' to edge 'to'
            value: length of the lane
        """
        lane_ids = self.get_turning_lanes_for_edge_pair(fr, to)
        lengths = dict( (id, self.lanes[id]['len']) for id in lane_ids)
        return lengths

    def compute_lane_angles_and_intermed_distances(self, lane_id):
        """
        Computes the following additional data for the lane with the passed id:
            * intermed_distances : List of floats
                Distances from the start of the lane to the start of the i-th segment
                of the lane shape
            * dir_vectors : List of pairs [float, float]
                The unit direction vector along the i-th segment of the lane shape
            * segment_angles: List of floats
                Angle the i-th segment of the lane shape makes with the positive vertical
                direction

        The lane shape is a list of pairs [[x0, y0], ..., [xN, yN]] of (x, y) coordinates
        that describes the geometric curve of the lane. 'The i-th segment' of the lane shape
        refers to the line segment from [x_{i-1}, y_{i-1}] to [xi, yi]
        """
        lane = self.lanes[lane_id]
        lane['intermed_distances'] = []
        lane['dir_vectors'] = []
        lane['segment_angles'] = []

        x0, y0 = lane['shape'][0]
        cmlt_distance = 0
        for x1, y1 in lane['shape'][1:]:
            # Degenerate case when the intermediate segment
            # has length 0 (these occur in the China networks)
            if (x1, y1) == (x0, y0):
                lane['intermed_distances'].append(cmlt_distance)
                lane['dir_vectors'].append(lane['dir_vectors'][-1])
                lane['segment_angles'].append(lane['segment_angles'][-1])
                continue

            # General case
            dir = [x1 - x0, y1 - y0]
            angle = find_angle(dir[0], dir[1])
            dD = math.sqrt(dir[0]*dir[0] + dir[1]*dir[1])
            dir = [ dir[0] / dD, dir[1] / dD ]

            lane['intermed_distances'].append(cmlt_distance)
            lane['dir_vectors'].append(dir)
            lane['segment_angles'].append(angle)
            cmlt_distance += dD
            x0, y0 = x1, y1

    def compute_all_lane_angles_and_intermed_distances(self):
        """
        Calls the 'compute_lane_angles_and_intermed_distances' for all lanes
        (including the lanes in the vertex turns)
        """
        for e in self.E.values():
            for lane_id in e['lanes'].keys():
                self.compute_lane_angles_and_intermed_distances(lane_id)
        for v in self.V.values():
            for turn in v['turns'].values():
                for lane_id in turn['lane_ids']:
                    self.compute_lane_angles_and_intermed_distances(lane_id)

    def lane_crds_to_absolute_crds(self, lane_id, d):
        """
        Finds the following attributes of a point at distance 'd' from the start
        of the lane 'lane_id':
            * x, y : Floats
                The x and y coordinates of the point
            * angle : Float
                The angle that the lane tangent vector makes with the positive y-direction
                at the point

        Precondition
        ------------
            The method 'compute_all_lane_angles_and_intermed_distances' is assumed to have
            been called before this method is used.
        """
        lane = self.lanes[lane_id]
        assert 0 <= d <= lane['len'], \
              f'The point at distance {d} from the start of the lane is not within the lane'

        # Multiplicative conversion factor from logical to geometric length. (The two may be
        # different in SUMO, see 'shape pos' (geometric) and 'length pos' (logical) in netedit.)
        conv = lane['shape_len'] / lane['len']

        # If the point is on the last segment of the lane shape,
        # return the data for the last segment
        if d >= lane['intermed_distances'][-1]:
            leftover_dist = d - lane['intermed_distances'][-1]
            x = lane['shape'][-2][0] + lane['dir_vectors'][-1][0] * leftover_dist * conv
            y = lane['shape'][-2][1] + lane['dir_vectors'][-1][1] * leftover_dist * conv
            angle = lane['segment_angles'][-1]
            return x, y, angle

        # Otherwise, binary search for the segment of the lane shape at distance 'd' from
        # the start of the lane
        start = 0
        end = len(lane['intermed_distances']) - 1
        while start <= end:
            mid = (start + end) // 2
            if d == lane['intermed_distances'][mid]:
                bin = mid
                break
            elif d < lane['intermed_distances'][mid]:
                if d >= lane['intermed_distances'][mid-1]:
                    bin = mid - 1
                    break
                else:
                    end = mid - 1
            elif d > lane['intermed_distances'][mid]:
                if d < lane['intermed_distances'][mid+1]:
                    bin = mid
                    break
                else:
                    start = mid + 1

        leftover_dist = d - lane['intermed_distances'][bin]
        x = lane['shape'][bin][0] + lane['dir_vectors'][bin][0] * leftover_dist * conv
        y = lane['shape'][bin][1] + lane['dir_vectors'][bin][1] * leftover_dist * conv
        angle = lane['segment_angles'][bin]

        return x, y, angle


    def find_detection_area_bdry(self, vertex_id, detection_range, only_straight=True):
        """
        Finds the lanes that are cut by the boundary of the detection area, and the (x, y)
        coordinates of the cut.

        Arguments
        ---------
            vertex_id : String
                Vertex with a detector
            detection_range : Float
                Range of detection on all approaches to the vertex
            only_straight : Boolean
                Whether only straight approaches are considered to lie within the detection
                area (whether or not turns are allowed)

        Returns
        -------
            Set of (x, y, angle)
                (x, y) are coordinates of the cut point on a lane cut by the boundary of the
                detection area, and angle is the angle the lane makes at that point with the
                vertical direction.
        """
        vertex = self.V[vertex_id]

        in_edges = self.V[vertex_id]['in_edges']
        cut_edges = set()
        cuts = set()
        for e in in_edges:
            upstream_routes = self.get_routes_upstream_to_edge(e, detection_range, only_straight)
            cut_edges |= set( (route['ids'][0], route['last_edge_cut']) for route in upstream_routes )

        for (edge_id, last_edge_cut) in cut_edges:
            last_edge_cut = max(0, last_edge_cut)
            for lane_id in self.E[edge_id]['lanes']:
                self.compute_lane_angles_and_intermed_distances(lane_id)
                cuts.add(self.lane_crds_to_absolute_crds(lane_id, last_edge_cut))

        return cuts

    def find_detector_polygon(self, lane_id, pos, length=4, width=2):
        """
        Finds the polygon of an induction loop detector given the lane_id
        and the relative position of the detector on the lane.

        Arguments
        ---------
            lane_id : String
                The network lane the detector lies on
            pos : Float
                The position of the centre of the detector from the start of
                the lane
            length, width : Floats, optional
                The dimensions of the detector rectangle

        Returns
        -------
            4-list of 2-lists, 2-list of 2-lists
                The coordinates of the four corners of the polygon,
                and the coordinates of the endpoints of the middle line
        """
        x, y, a = self.lane_crds_to_absolute_crds(lane_id, pos)
        a = math.radians(a)
        # Unit direction vector parallel to the lane
        D = [math.sin(a), math.cos(a)]
        # Unit normal (cw from D)
        N = [D[1], -D[0]]
        return ([[x + 2*D[0] - N[0], y + 2*D[1] - N[1]],
                 [x + 2*D[0] + N[0], y + 2*D[1] + N[1]],
                 [x - 2*D[0] + N[0], y - 2*D[1] + N[1]],
                 [x - 2*D[0] - N[0], y - 2*D[1] - N[1]]],
                [[x - N[0], y - N[1]], [x + N[0], y + N[1]]])



#TODO: Convert to just functions
    def __repr__(self):
        return json.dumps(self.to_JSON(), indent=2)

    def to_JSON(self):
        return {'V': self.V,
                'E': self.E,
                'lanes': self.lanes,
                'detectors': self.detectors,
                'bbox': self.bbox}


    def to_XML_netfile(self):
        # XML namespaces, to set up the root <net> node
        xsi = 'http://www.w3.org/2001/XMLSchema-instance'
        nsmap = {'xsi': xsi}
        schema_location = etree.QName('%s' % (xsi), 'noNamespaceSchemaLocation')
        net_attrs = {'version': '1.1',
                      schema_location: 'http://sumo.dlr.de/xsd/net_file.xsd'}
        E = ElementMaker(nsmap=nsmap)
        net_XML_node = E.net(net_attrs) # the root <net> node is now initialized

        # Move the network so that its bottom-left coordinate is at (0, 0)
        dX = -self.bbox['x0']
        dY = -self.bbox['y0']

        internal_edges = []
        edges = []
        junctions = []
        connections = []
        connection_ends = []

        # Adding edges
        for edge_id, edge in self.E.items():
            edge_attrs = {'id': edge_id,
                          'from': edge['from'],
                          'to': edge['to'],
                          'length': str(edge['middle_lane_length']),
                          'shape': shape_str(
                              translate_shape(edge['middle_lane_shape'], dX, dY))}

            lanes = []
            for idx, lane in enumerate(edge['lanes']):
                lane_id = f'{edge_id}_{idx}'
                lane_attrs = {'id': lane_id,
                              'index': str(idx),
                              'speed': str(lane['speed_limit']),
                              'length': str(lane['len']),
                              'shape': shape_str(
                                  translate_shape(lane['shape'], dX, dY))}
                lanes.append(lane_attrs)
            edges.append((edge_attrs, lanes))

        # Adding vertices (junctions), internal edges, and connections
        for vtx_id, vtx in self.V.items():
            # Internal edges and lanes
            internal_edge_id = f':{str(vtx_id)}'
            internal_edge_attrs = {'id': internal_edge_id,
                                   'function': 'internal'}
            internal_lanes = []
            internal_lane_counter = 0
            for turn in vtx['turns'].values():
                from_edge = turn['from_edge']
                to_edge = turn['to_edge']
                for inc, out in zip(turn['from_lane_indices'], turn['to_lane_indices']):
                    via = f'{internal_edge_id}_{internal_lane_counter}'
                    lane_attrs = {'id': via,
                                  'index': str(internal_lane_counter),
                                  'speed': str(turn['speed_limit']),
                                  'length': str(turn['len']),
                                  'shape': shape_str(
                                      translate_shape(turn['shape'], dX, dY))}
                    internal_lanes.append(lane_attrs)

                    connection_attrs = {'from': from_edge,
                                        'to': to_edge,
                                        'fromLane': str(inc),
                                        'toLane': str(out),
                                        'via': via}

                    connections.append(connection_attrs)
                    connection_end_attrs = {'from': internal_edge_id,
                                            'to': to_edge,
                                            'fromLane': str(internal_lane_counter),
                                            'toLane': str(out)}
                    connection_ends.append(connection_end_attrs)

                    internal_lane_counter += 1

            internal_edges.append((internal_edge_attrs, internal_lanes))

            # Junction
            inc_lanes = []
            for edge_id in self.V[vtx_id]['in_edges']:
                inc_lanes.extend([lane['id'] for lane in self.E[edge_id]['lanes']])
            inc_lanes = ' '.join(inc_lanes)
            int_lanes = ' '.join(lane['id'] for lane in internal_lanes)

            jnct_attrs = {'id': vtx_id,
                          'type': str(vtx['type']),
                          'x': str(vtx['x'] + dX),
                          'y': str(vtx['y'] + dY),
                          'shape': shape_str(
                              translate_shape(vtx['polygon'], dX, dY)),
                          'intLanes': int_lanes,
                          'incLanes': inc_lanes}
            junctions.append(jnct_attrs)


        # Create the XML nodes
        convx1 = self.bbox['x1'] - self.bbox['x0']
        convy1 = self.bbox['y1'] - self.bbox['y0']
        location_attrs = {'netOffset': '0.00,0.00',
                          'convBoundary': f'0.00,0.00,{convx1},{convy1}',
                          'origBoundary': f'{self.bbox["x0"]},{self.bbox["y0"]},{self.bbox["x1"]},{self.bbox["y1"]}',
                          'projParameter': '!'}
        location_XML_node = etree.SubElement(net_XML_node, 'location', location_attrs)

        for internal_edge_attrs, lanes in internal_edges:
            internal_edge_XML_node = etree.SubElement(net_XML_node, 'edge', internal_edge_attrs)
            for internal_lane_attrs in lanes:
                internal_lane_XML_node = etree.SubElement(internal_edge_XML_node, 'lane', internal_lane_attrs)

        for edge_attrs, lanes in edges:
            edge_XML_node = etree.SubElement(net_XML_node, 'edge', edge_attrs)
            for lane_attrs in lanes:
                lane_XML_node = etree.SubElement(edge_XML_node, 'lane', lane_attrs)

        for jnct_attrs in junctions:
            jnct_XML_node = etree.SubElement(net_XML_node, 'junction', jnct_attrs)

        for cnctn_attrs in connections:
           cnctn_XML_node = etree.SubElement(net_XML_node, 'connection', cnctn_attrs)

        for cnctn_end_attrs in connection_ends:
           cnctn_end_XML_node = etree.SubElement(net_XML_node, 'connection', cnctn_end_attrs)

        netfile = etree.ElementTree(net_XML_node)
        netfile.write('./test.xml', pretty_print=True)
