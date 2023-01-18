import numpy as np
import geopandas as gpd
import pickle
from src.congestion_mining.data import TrafficData


class AimsunData(TrafficData):

    def __init__(self, path, shapefile, data_file):
        super().__init__()

        self.df = gpd.read_file(path+shapefile)

        with open(path+data_file, 'rb') as pickle_file:
            content = pickle.load(pickle_file)

        self.links = content.links
        self.data = content.data
        self.flow = self.data[0]
        self.speeds = self.data[1]

        spd_limits = []
        lengths = []
        for link in self.links:
            spd_limits.append(self.df.loc[self.df['linkID'] == link, 'speed'].item())
            lengths.append(self.df.loc[self.df['linkID'] == link, 'length'].item())

        self.spd_limits = np.array(spd_limits)
        self.lengths = np.array(lengths)

        self.congestion = np.zeros(self.speeds.shape)
        self.rel_speed = np.zeros(self.speeds.shape)
        for i in range(len(self.speeds)):
            self.rel_speed[i] = self.speeds[i] / self.spd_limits[i]
            self.congestion[i] = np.where(self.speeds[i] / self.spd_limits[i] < 0.5, 1, 0)
