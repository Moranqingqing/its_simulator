import fiona
from shapely.geometry import LineString
from shapely.geometry import Point
from haversine import haversine

from src.network.network_model import NetworkModel


class HERENetwork(NetworkModel):
    def __init__(self, shp_path):
        super().__init__()
        with fiona.open(shp_path) as source:
            link_geometry = {}
            for road in source:
                self.link_ids.append(str(road['properties']['LINK_ID']) + "F")
                self.link_ids.append(str(road['properties']['LINK_ID']) + "T")
                link_geometry[road['properties']['LINK_ID']] = road['geometry']

            # not sure if there are intersections between start and end, if so this whole thing breaks
            for link, line in link_geometry.items():
                # first and last point on the line
                start = Point(line['coordinates'][0])
                end = Point(line['coordinates'][-1])
                ls = LineString(line['coordinates'])

                # add a ~50m buffer
                # unit is in longitude/latitude so this may be very off
                start = start.buffer(0.00005)
                end = end.buffer(0.00005)

                # the HERE data uses a system where start -> end is #########F and end -> start is #########T
                # ######### is the link ID attribute
                from_link = str(link) + "F"
                to_link = str(link) + "T"
                self.upstream[from_link] = {'left': [], 'straight': [], 'right': []}
                self.upstream[to_link] = {'left': [], 'straight': [], 'right': []}
                self.downstream[from_link] = {'left': [], 'straight': [], 'right': []}
                self.downstream[to_link] = {'left': [], 'straight': [], 'right': []}
                coordinates = list(ls.coords)
                self.length[from_link] = haversine(coordinates[0], coordinates[1])
                self.length[to_link] = haversine(coordinates[0], coordinates[1])

                # check every other road and see if it's within 50m of start and end
                # by checking if the line intersects with our buffer
                for link2, line2 in link_geometry.items():
                    line2 = LineString(line2['coordinates'])
                    if link != link2 and start.intersects(line2):
                        self.upstream[from_link]['straight'].append(str(link2) + "F")
                        self.upstream[from_link]['straight'].append(str(link2) + "T")
                        self.downstream[to_link]['straight'].append(str(link2) + "F")
                        self.downstream[to_link]['straight'].append(str(link2) + "T")
                    if link != link2 and end.intersects(line2):
                        self.upstream[to_link]['straight'].append(str(link2) + "F")
                        self.upstream[to_link]['straight'].append(str(link2) + "T")
                        self.downstream[from_link]['straight'].append(str(link2) + "F")
                        self.downstream[from_link]['straight'].append(str(link2) + "T")
