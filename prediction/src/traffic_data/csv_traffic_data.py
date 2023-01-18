import csv

import numpy as np
import xarray as xr

from src.traffic_data.traffic_data import TrafficData


class CSVTrafficData(TrafficData):
    """
    Aimsun data in CSV format, this class should not be used
    """

    def __init__(self, csv_path, network=None):
        super().__init__(network=network)

        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            ncols = len(header)

            # first 2 columns are id and timestamp
            column_names = header[2:]
            self.features = column_names
            for row in reader:
                if row[0] not in self.links:
                    self.links.append(row[0])
                if row[1] not in self.timestamps:
                    self.timestamps.append(row[1])
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=range(2, ncols))
        data = data.reshape((len(self.links), len(self.timestamps), len(self.features)))
        # (features, links, times)
        self.data = np.transpose(data, (2, 0, 1))
        for i in range(len(self.links)):
            assert self.links[i] == self.network.link_ids[i]

        self.data = xr.DataArray(self.data, dims=("feature", "link", "time"),
                                 coords={"feature": self.features, "link": self.links, "time": self.timestamps})
