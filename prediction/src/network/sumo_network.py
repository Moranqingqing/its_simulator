from lxml import etree

from src.network.network_model import NetworkModel


# SUMO network is specified by an XML file
class SUMONetwork(NetworkModel):
    def __init__(self, xml_path):
        super().__init__()
        tree = etree.parse(xml_path)
        net = tree.getroot()
        # get the id for all edges
        for link in net:
            if link.tag == 'edge':
                link_id = link.get('id')
                self.link_ids.append(link_id)
                self.downstream[link_id] = {'left': [], 'straight': [], 'right': []}
                self.upstream[link_id] = {'left': [], 'straight': [], 'right': []}

        # for each connection, mark the corresponding links to upstream and downstream of each other
        # only supports the 3 main turn types, can add more later
        for connection in net:
            if connection.tag == 'connection':
                from_link = connection.get('from')
                to_link = connection.get('to')

                if connection.get('dir') == 'l':
                    direction = 'left'
                elif connection.get('dir') == 'r':
                    direction = 'right'
                else:
                    direction = 'straight'

                # connection is marked for every lane, check before adding to avoid duplicates
                if to_link not in self.downstream[from_link][direction]:
                    self.downstream[from_link][direction].append(to_link)
                if from_link not in self.upstream[to_link][direction]:
                    self.upstream[to_link][direction].append(from_link)
