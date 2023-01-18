import sqlite3

import numpy as np
import xarray as xr

from src.traffic_data.traffic_data import TrafficData


class AimsunTrafficData(TrafficData):
    def __init__(self, sqlite_path, run_len=0, network=None, intersection=False):
        """
        Traffic data for Aimsun generated database files

        Parameters
        ----------
        sqlite_path
            Path to the SQLite database file
        run_len
            The length of each run, in number of timesteps
        network
            The NetworkModel this data is associated with
        intersection
            Whether to include intersection features
        """
        super().__init__(network=network, run_len=run_len)
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        self.links = self.network.link_ids
        cur.execute("SELECT did, ent FROM MISECT WHERE ent<>0 AND ent<=? GROUP BY did, ent ORDER BY did, ent",
                    (run_len,))
        self.timestamps = [str(i[0]) + '_' + str(i[1]) for i in cur.fetchall()]
        self.features = ['flow', 'speed']
        network_data = []

        # get phase history for each intersection in subnetwork
        phase_dict = {}
        ids = []
        if intersection:
            cur.execute('''select * from CONTROLPHASE''')
            cols = list(map(lambda x: x[0], cur.description))
            res = cur.fetchall()
            res = [dict(zip(cols, r)) for r in res]
            ids = list(set([i['node_id'] for i in res]))

            for i in ids:
                cur.execute("SELECT did, ent, phase, active_time_percentage FROM CONTROLPHASE WHERE node_id=? AND ent<>0 AND ent<=? "
                            "ORDER BY did, ent, phase", (i, run_len))
                phases = cur.fetchall()
                runs = {run: index for index, run in enumerate(list(set([i[0] for i in phases])))}
                num_runs = len(set([i[0] for i in phases]))
                num_phases = max([i[2] for i in phases])
                active_time = np.zeros((num_runs * run_len, num_phases))
                for j in phases:
                    run = runs[j[0]]
                    active_time[run * run_len + j[1] - 1, j[2] - 1] = j[3]
                phase_dict[i] = active_time

        # number of features in each intersection may be different, use this to keep track
        max_features = 0
        for link in self.links:
            link_data = cur.execute("SELECT flow, spdh "
                                    "FROM MISECT WHERE oid=? AND sid=0 AND ent<>0 AND ent<=? ORDER BY did, ent",
                                    (link, run_len))
            link_data = np.array(list(link_data))
            link_data[link_data == -1] = self.network.speed_limit[link]

            # grab intersection features from upstream and downstream nodes
            if intersection:
                upstream_node = self.network.upstream_node[link]
                downstream_node = self.network.downstream_node[link]

                # upstream
                if upstream_node in ids:
                    active_time = phase_dict[upstream_node]
                    link_data = np.concatenate([link_data, active_time], axis=1)

                # downstream
                if downstream_node in ids:
                    active_time = phase_dict[downstream_node]
                    link_data = np.concatenate([link_data, active_time], axis=1)

            # update max features
            if link_data.shape[1] > max_features:
                max_features = link_data.shape[1]

            network_data.append(link_data)

        # make the number of features the same across all links by padding 0's
        network_data = [np.concatenate([link, np.zeros((link.shape[0], max_features - link.shape[1]))], axis=1)
                        for link in network_data]
        for _ in range(max_features - 2):
            self.features.append('intersection')

        self.data = np.array(network_data)
        self.data = np.transpose(self.data, (2, 0, 1))

        self.data = xr.DataArray(self.data, dims=("feature", "link", "time"),
                                 coords={"feature": self.features, "link": self.links, "time": self.timestamps})
