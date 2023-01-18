import json

import numpy as np
import xarray as xr

from src.traffic_data.traffic_data import TrafficData


class JSONTrafficData(TrafficData):
    """
    Traffic data in JSON format, this class should not be used
    """

    def __init__(self, json_path, network=None):
        super().__init__(network=network)

        f = open(json_path, 'r')
        file_data = json.load(f)
        data = []
        for feature, link_data in file_data.items():
            self.features.append(feature)
            link_speeds = []
            for link, speeds in link_data.items():
                if link not in self.links:
                    self.links.append(link)
                link_speeds.append(speeds)
            data.append(link_speeds)
        self.data = np.array(data)
        self.timestamps = list(range(self.data.shape[2]))
        f.close()

        self.data = xr.DataArray(self.data, dims=("feature", "link", "time"),
                                 coords={"feature": self.features, "link": self.links, "time": self.timestamps})
