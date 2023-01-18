import os, sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import numpy as np
import pandas as pd
import sumolib
from src.congestion_mining.data import TrafficData


class SUMOTrafficData(TrafficData):

    def __init__(self, net, data_folder, sumo_config_file, model='binary'):
        super().__init__()

        self.path = net.path
        self.data_folder = data_folder
        self.sumo_config_file = self.path + sumo_config_file

        files = []
        for f in sumolib.xml.parse(self.sumo_config_file, ['net-file', 'route-files', 'additional-files']):
            files.append(f.value)

        self.net_file = self.path + files[0]
        self.route_file = self.path + files[1]
        self.additional_file = self.path + files[2]

        for f in sumolib.xml.parse_fast(self.additional_file, 'edgeData', ['file', 'freq']):
            freq = int(f.freq)
            data_file = f.file
        self.freq = freq
        self.data_file = self.path + self.data_folder + data_file

        for interval, edge in sumolib.xml.parse_fast_nested(self.data_file, 'interval', ['begin', 'end'],
                                                            'edge', ['id']):
            if interval.end not in self.timestamps:
                self.timestamps.append(interval.end)
            if edge.id not in self.links:
                self.links.append(edge.id)

        time = self.freq
        self.shape = (len(self.links), len(self.timestamps))
        self.speeds = np.zeros(self.shape)
        self.speeds[:] = np.nan
        self.spd_limits = np.zeros(len(self.links))
        travel_time = np.zeros(self.shape)
        travel_time[:] = np.nan
        entered = np.zeros(self.shape)
        entered[:] = np.nan
        left = np.zeros(self.shape)
        left[:] = np.nan

        for interval, edge in sumolib.xml.parse_fast_nested(self.data_file, 'interval', ['begin', 'end'],
                                                            'edge', ['id', 'traveltime', 'speed', 'entered', 'left']):
            time = interval.end
            t_ind = self.timestamps.index(time)
            ind = self.links.index(edge.id)

            self.speeds[ind][t_ind] = edge.speed
            self.spd_limits[ind] = net.spd_lims[edge.id]
            travel_time[ind][t_ind] = edge.traveltime
            entered[ind][t_ind] = edge.entered
            left[ind][t_ind] = edge.left

        self.travel_time = travel_time
        self.entered = entered
        self.left = left

        for link in self.speeds:
            link[np.isnan(link)] = 0

        self.congestion = np.zeros(self.speeds.shape)
        self.rel_speed = np.zeros(self.speeds.shape)
        for i in range(len(self.speeds)):
            self.rel_speed[i] = self.speeds[i] / self.spd_limits[i]
            self.congestion[i] = np.where(self.speeds[i] / self.spd_limits[i] < 0.5, 1, 0)
