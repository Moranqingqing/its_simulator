import glob
import pickle

import fiona
import numpy as np
import pandas as pd
import xarray as xr

from src.traffic_data.traffic_data import TrafficData


class HERETrafficData(TrafficData):
    def __init__(self, network, folder_path, shp_path, out_path):
        super().__init__(network)
        non_res = self.get_non_residential(shp_path)
        traffic_data_list = []
        csv_files = glob.glob(folder_path + "*.csv")
        for csv_path in csv_files:
            all_data = pd.read_csv(csv_path, usecols=[0, 1, 6, 7], parse_dates=[1],
                                   infer_datetime_format=True, chunksize=10000 * 288 * 31)
            for i, data in enumerate(all_data):
                self.features = ['count', 'speed']
                links = list(dict.fromkeys(list(data.iloc[:, 0])))
                links = [link for link in links if non_res[int(link[:-1])]]
                self.links.extend(links)
                print(len(self.links))
                self.timestamps = list(dict.fromkeys(list(data.iloc[:, 1])))
                # get rid of residential streets
                data = data[data.apply(lambda x: non_res[int(x['LINK-DIR'][:-1])], axis=1)]
                data = data.iloc[:, 2:4].values
                data = data.reshape((len(links), len(self.timestamps), len(self.features)))
                data = np.transpose(data, (2, 0, 1))  # [features, links, timestamps]
                traffic_data_list.append(data)
            self.data = np.concatenate(traffic_data_list, axis=1)

        self.data = xr.DataArray(self.data, dims=("feature", "link", "time"),
                                 coords={"feature": self.features, "link": self.links, "time": self.timestamps})

        f = open(out_path, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def get_non_residential(shapefile='data/HERE/shapefile/Streets.shp'):
        output = {}
        source = fiona.open(shapefile)
        for row in source:
            if row['properties']['LINK_ID'] not in output:
                if row['properties']['FUNC_CLASS'] != '5':
                    output[row['properties']['LINK_ID']] = True
                else:
                    output[row['properties']['LINK_ID']] = False
        return output
