from src.normalizer import *


class TrafficData:
    """
    The base class for all traffic data

    Parameters
    ----------
    network : NetworkModel
        The network model that this traffic data uses
    run_len : int, optional
        The length of each simulation run if this data contains multiple runs
    normalize : str or None
        the method of traffic data normalization. currently only supports 'standard' or None.
        'standard': use standard deviation and mean to normalize data.
        None: no normalization.

    Attributes
    ----------
    features : list of str
    links : list of str
    timestamps : list of str or list of datetime
    network : NetworkModel
    data : xr.DataArray
        3D array representation of data, axis 0 is features, axis 1 is links, axis 2 is time
    train_data : xr.DataArray
        3D array, split from data in time axis
    val_data : xr.DataArray
        3D array, split from data in time axis
    test_data : xr.DataArray
        3D array, split from data in time axis
    train_timestamps : list of str or list of datetime
        list of timestamps, subset of timestamps
    val_timestamps : list of str or list of datetime
        list of timestamps, subset of timestamps
    test_timestamps : list of str or list of datetime
        list of timestamps, subset of timestamps
    run_len : int
        The length of each simulation run if this data contains multiple runs
    normalize : str or None
        the method of traffic data normalization. currently only supports 'standard' or None.
        'standard': use standard deviation and mean to normalize data.
        None: no normalization.
    normalizer : dict
        {key: value} = {feature: instance of Normalizer class fed by data}
    """

    def __init__(self, network=None, run_len=None, normalize=None):
        self.network = network
        self.data = None

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.features = []
        self.links = []
        self.timestamps = []
        self.train_timestamps = []
        self.val_timestamps = []
        self.test_timestamps = []

        self.normalize = normalize
        self.normalizer = {}

        self.run_len = run_len

    def split(self, val_pct=0.0, test_pct=0.0):
        """
        Splits the dataset in the time axis into train, validation, and test set

        Parameters
        ----------
        val_pct : float
            percentage of data to include in the validation set
        test_pct : float
            percentage of data to include in the test set
        """
        num_val = int(val_pct * len(self.timestamps))
        num_test = int(test_pct * len(self.timestamps))
        num_train = len(self.timestamps) - num_val - num_test

        self.train_data = self.data[:, :, :num_train]
        self.val_data = self.data[:, :, num_train:num_train + num_val]
        self.test_data = self.data[:, :, num_train + num_val:]

        self.train_timestamps = self.timestamps[:num_train]
        self.val_timestamps = self.timestamps[num_train:num_train + num_val]
        self.test_timestamps = self.timestamps[num_train + num_val:]

    def split_by_timestamp(self, train_timestamps, val_timestamps, test_timestamps):
        """
        Splits the dataset in the time axis into train, validation, and test set according to timestamps

        Parameters
        ----------
        train_timestamps : list
            list of timestamps to include in the training set
        val_timestamps : list
            list of timestamps to include in the validation set
        test_timestamps : list
            list of timestamps to include in the test set
        """
        self.train_data = self.data[:, :, np.where(np.isin(self.timestamps, train_timestamps))[0]]
        self.val_data = self.data[:, :, np.where(np.isin(self.timestamps, val_timestamps))[0]]
        self.test_data = self.data[:, :, np.where(np.isin(self.timestamps, test_timestamps))[0]]

        self.train_timestamps = train_timestamps
        self.val_timestamps = val_timestamps
        self.test_timestamps = test_timestamps

    def init_normalizer(self):
        if self.normalize == 'standard':
            self.normalizer = {feature: StandardNormalizer(self.data.sel(feature=feature).data) for feature in self.data.feature.data}
        else:
            self.normalizer = {feature: NoneNormalizer(self.data.sel(feature=feature).data) for feature in self.data.feature.data}

    def set_normalize(self, normalize):
        self.normalize = normalize
        self.init_normalizer()

    def transform(self):
        for feature in self.data.feature.data:
            self.data.loc[feature] = self.normalizer[feature].transform()

    def inverse_transform(self):
        for feature in self.data.feature.data:
            self.data.loc[feature] = self.normalizer[feature].inverse_transform()

    # def split_random(self, test_pct):
    #     """
    #     Splits the data for top level nested CV
    #     Parameters
    #     ----------
    #     test_pct : float
    #         percentage of data to include in the test set
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     # simulation data with multiple runs
    #     if self.run_len:
    #         num_runs = int(len(self.timestamps) / self.run_len)
    #         test_runs = random.sample(range(num_runs), int(test_pct * num_runs))
    #         test_indices = np.array(
    #             [list(range(self.run_len * i, self.run_len * (i + 1))) for i in test_runs]).flatten()
    #         test_indices = sorted(test_indices)
    #         train_indices = [i for i in range(len(self.timestamps)) if i not in test_indices]
    #     # real data with 1 run
    #     else:
    #         num_test = int(len(self.timestamps) * test_pct)
    #         start_index = random.randint(0, len(self.timestamps) - num_test)
    #         test_indices = range(start_index, start_index + num_test)
    #         train_indices = [i for i in range(len(self.timestamps)) if i not in test_indices]
    #
    #     self.test_timestamps = [self.timestamps[i] for i in test_indices]
    #     self.val_timestamps = []
    #     self.train_timestamps = [self.timestamps[i] for i in train_indices]
    #
    #     self.test_data = self.data[:, :, test_indices]
    #     self.val_data = None
    #     self.train_data = self.data[:, :, train_indices]
