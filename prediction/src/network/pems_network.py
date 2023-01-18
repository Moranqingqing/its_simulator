import csv

from src.network.network_model import NetworkModel


class PeMSNetwork(NetworkModel):
    def __init__(self, pems_csv, num_nodes):
        super().__init__()
        self.link_ids = [i for i in range(num_nodes)]
        self.downstream = {i: {'left': [], 'straight': [], 'right': []} for i in range(num_nodes)}
        self.upstream = {i: {'left': [], 'straight': [], 'right': []} for i in range(num_nodes)}

        with open(pems_csv, "r") as f_d:
            f_d.readline()  # skip table head
            reader = csv.reader(f_d)
            for item in reader:
                if len(item) != 3:
                    continue
                i, j, distance = int(item[0]), int(item[1]), float(item[2])

                self.downstream[i]['straight'].append(j)
                self.upstream[j]['straight'].append(i)
