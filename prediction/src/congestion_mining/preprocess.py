import geopandas as gpd
from haversine import haversine
from shapely.geometry import LineString
import numpy as np
import pickle

col_mapping = {'speed': ['speed', 'Speed', 'SPEED', 'speedatt', 'SPD_LIM'],
               'linkID': ['linkID', 'id', 'LINK_ID']}

class ShapefileProcessing:
    def __init__(self, path, shapefile, name, lat_lon=True):

        self.path = path
        self.shapefile = shapefile
        self.df = gpd.read_file(self.path + self.shapefile)

        if 'fnode' in self.df.columns:
            self.df['fnode'] = self.df['fnode'].astype(str)
        if 'tnode' in self.df.columns:
            self.df['tnode'] = self.df['tnode'].astype(str)

        if 'DIR_TRAVEL' in self.df.columns:

            # create new df to shift coordinates in order to display both F and T directions
            df_T = self.df[self.df['DIR_TRAVEL'] == 'B']
            df_T.loc[:, 'geometry'] = df_T.geometry.apply(lambda row: self.update_coords(row, shift=False))
            df_T.loc[:, 'DIR_TRAVEL'] = 'T'

            self.df.loc[self.df['DIR_TRAVEL'] == 'B', 'DIR_TRAVEL'] = 'F'
            self.df = self.df.append(df_T)
            self.df['linkID'] = self.df['LINK_ID'].astype(str) + self.df['DIR_TRAVEL']
            self.df['speed'] = np.where(self.df['DIR_TRAVEL'] == 'T', self.df['TO_SPD_LIM'], self.df['FR_SPD_LIM'])

        if 'length' not in self.df.columns:
            self.df['length'] = self.df['geometry'].apply(lambda geom: self.get_length(geom, lat_lon))

        self.df = self.df.rename(mapper=self.get_col, axis=1)
        self.df['linkID'] = self.df['linkID'].astype(int).astype(str)

        output_file = self.path + name
        self.df.to_file(output_file)

    @staticmethod
    def get_col(column):

        for key, cols in col_mapping.items():
            if column in cols:
                return key
        return column

    @staticmethod
    def get_length(geom, lat_lon):

        if lat_lon:
            coordinates = list(geom.coords)
            length = haversine(coordinates[0], coordinates[1])
        else:
            length = geom.length

        return length

    @staticmethod
    def update_coords(row, shift):

        x_off = 0.0003
        y_off = 0.0002

        if shift:
            coords = []
            for coord in list(row.coords):
                x, y = coord
                x_shift = x + x_off
                y_shift = y + y_off
                coord = tuple((x_shift, y_shift))
                coords.append(coord)
        else:
            coords = list(row.coords)
        coords.reverse()

        return LineString(coords)
