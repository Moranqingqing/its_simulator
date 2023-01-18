from src.congestion_mining.data import TrafficData

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString
import pickle
from haversine import haversine

class HERETrafficData(TrafficData):

    def __init__(self, shapefile, speedfile, data_path='data/HERE/', usecols=None, preprocess_shp=False):
        super().__init__()

        self.shapefile = shapefile
        self.speedfile = speedfile
        self.data_path = data_path

        rpath = self.data_path + self.shapefile
        if usecols:
            self.df = gpd.read_file(rpath).filter(usecols)
        else:
            self.df = gpd.read_file(rpath)

        fpath = self.data_path + self.speedfile
        f = open(fpath, 'rb')
        self.speed_data = pickle.load(f)
        f.close()

        self.counts = self.speed_data.data[0]
        self.speeds = self.speed_data.data[1]
        self.links = self.speed_data.links
        self.timestamps = self.speed_data.timestamps

        self.df = self.df[self.df['linkID'].isin(self.links)]

        spd_limits = []
        lengths = []
        for link in self.links:
            spd_limits.append(self.df.loc[self.df['linkID'] == link, 'speed'].item())
            lengths.append(self.df.loc[self.df['linkID'] == link, 'length'].item())

        self.spd_limits = np.array(spd_limits)
        self.lengths = np.array(lengths)

        congestion = np.zeros(self.speeds.shape)
        for i in range(len(self.speeds)):
            rel_speed = self.speeds[i] / self.spd_limits[i]
            congestion[i] = np.where(rel_speed < 0.5, 1, 0)

        self.congestion = congestion

    def export_shp(self, name):

        output_file = self.data_path + name
        self.df.to_file(output_file)