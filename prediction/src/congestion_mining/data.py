import pickle
import numpy as np
import pandas as pd

class TrafficData:

    """
    The base class for all traffic data

    Parameters
    ----------

    Attributes
    ----------
    links : list of str
    timestamps : list of str or list of datetime
    data : numpy.array
        3D array representation of data, axis 0 is features, axis 1 is links, axis 2 is time
    """

    def __init__(self):
        self.links = []
        self.timestamps = []
        self.counts = []
        self.data = None
        self.speeds = None
        self.spd_limits = []
        self.rel_speeds = []
        self.congestion = []

    @staticmethod
    def get_EFD_bins(data, number_of_bins):
        _, bins = pd.qcut(data.flatten(), q=number_of_bins, retbins=True, duplicates='drop')
        return bins

    @staticmethod
    def get_IQR_bins(data):
        q2 = np.quantile(data, 0.25)
        q4 = np.quantile(data, 0.75)
        return [data.min(), q2, q4, data.max()]

    def discretize(self, method, **kwargs):
        self.congestion = np.zeros(self.speeds.shape)
        self.rel_speeds = np.zeros(self.speeds.shape)
        for i in range(len(self.speeds)):
            self.rel_speeds[i] = self.speeds[i] / self.spd_limits[i]

        if method == 'threshold':
            for i in range(len(self.speeds)):
                self.congestion[i] = np.where(self.rel_speeds[i] < kwargs['C'], 1, 0)

        elif method == 'median':
            median = np.median(self.rel_speeds)
            self.congestion = np.where(self.rel_speeds < median, 1, 0)

        elif method == 'EWD':
            bin_edges = np.histogram_bin_edges(self.rel_speeds, kwargs['bins'])
            self.congestion = np.digitize(self.rel_speeds, bin_edges)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)


