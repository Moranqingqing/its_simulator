import json


class NetworkModel:
    """
    The class that represents a traffic network

    Attributes
    ----------
    link_ids : list
        list of all link IDs
    downstream : dict
        maps each link ID to the IDs of downstream links
    upstream : dict
        maps each link ID to the IDs of upstream links
    name : dict
        maps each link ID to its road name
    length : dict
        maps each link ID to its length
    """

    def __init__(self):
        self.link_ids = []
        self.downstream = {}
        self.upstream = {}
        self.name = {}
        self.length = {}
        self.speed_limit = {}

    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump([self.link_ids, self.downstream, self.upstream, self.name], f)

    def load(self, file_path):
        with open(file_path, 'r') as f:
            self.link_ids, self.downstream, self.upstream, self.name = json.load(f)
        return self
