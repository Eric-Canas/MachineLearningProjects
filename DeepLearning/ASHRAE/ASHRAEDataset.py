from torch.utils.data import Dataset
import numpy as np
import os
import re

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('max_columns', 150)

WEATHER_KEYS_IDX = [0, 8, 11]
WEATHER_VALUES_IDX = [1, 3, 5, 6, 7] #Erasing 2: Cloud Coverage, 4: Precipitation Depth 1hr

MONTHS = 12
DAYS = 7
HOURS = 24
#DF indexes
METER_IDX = 1
Y_IDX = 2
DF_MONTH_IDX = 4
DF_DAYWEEK_IDX = 5
DF_HOUR_IDX = 6
#Building data Indexes
PRIMARY_USE_IDX = 2
#Lengths
METER_LEN = 4
DF_DATA_LEN = METER_LEN+MONTHS+DAYS+HOURS
TYPES_LEN = 16
BUILDING_DATA_LEN = TYPES_LEN+3-2 #-2 for erasing year built and floor count
WEATHER_DATA_LEN = len(WEATHER_VALUES_IDX)
INPUT_LEN = DF_DATA_LEN+BUILDING_DATA_LEN+WEATHER_DATA_LEN



IDX_GROUPS = (range(0,METER_LEN), range(METER_LEN,METER_LEN+MONTHS), range(METER_LEN+MONTHS, METER_LEN+MONTHS+DAYS),
              range(METER_LEN+MONTHS+DAYS, DF_DATA_LEN), range(DF_DATA_LEN,DF_DATA_LEN+TYPES_LEN),
              DF_DATA_LEN+TYPES_LEN, DF_DATA_LEN+TYPES_LEN+1, DF_DATA_LEN+TYPES_LEN+2,
              DF_DATA_LEN+BUILDING_DATA_LEN, DF_DATA_LEN+BUILDING_DATA_LEN+1,DF_DATA_LEN+BUILDING_DATA_LEN+2,
              DF_DATA_LEN+BUILDING_DATA_LEN+3, DF_DATA_LEN+BUILDING_DATA_LEN+4, DF_DATA_LEN+BUILDING_DATA_LEN+5,
              DF_DATA_LEN+BUILDING_DATA_LEN+6)

BUILDING_DATA_COLS_TO_STANDARIZE = ['square_feet', 'year_built', 'floor_count']
WEATHER_COLS_TO_STANDARIZE = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
                              'sea_level_pressure', 'wind_direction', 'wind_speed']

class ASHRAEDataset(Dataset):
    def __init__(self, root = './Data', charge_train=True, charge_test=False, normalize=True, erase_nans=False):

        building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))
        building_meta_df['primary_use'] = np.unique(building_meta_df.to_numpy()[:, 2], return_inverse=True)[1]
        idx = [building_meta_df.columns.get_loc(col) for col in BUILDING_DATA_COLS_TO_STANDARIZE]
        self.building_meta_df = building_meta_df.to_numpy().astype(np.float32)
        if not erase_nans:
            self.building_meta_df = transform_nans(data=self.building_meta_df, operation='median')
        self.building_meta_df = standarize(data=self.building_meta_df, idx=idx)

        if charge_train:
            train_df = pd.read_csv(os.path.join(root,'train.csv'))
            # We save the time as day and hour in separated fields, because hour and season are better descriptors
            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            train_df['day_year'] = train_df["timestamp"].dt.dayofyear
            train_df['month'] = train_df["timestamp"].dt.month
            train_df['day'] = train_df["timestamp"].dt.dayofweek
            train_df['hour'] = train_df["timestamp"].dt.hour
            del train_df['timestamp']

            weather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))
            weather_train_df["timestamp"] = pd.to_datetime(weather_train_df["timestamp"])
            #We save the time as day and hour in separated fields, because hour and season are better descriptors
            weather_train_df['day_year'] = weather_train_df["timestamp"].dt.dayofyear
            weather_train_df['month'] = weather_train_df["timestamp"].dt.month
            weather_train_df['day'] = weather_train_df["timestamp"].dt.dayofweek
            weather_train_df['hour'] = weather_train_df["timestamp"].dt.hour
            del weather_train_df['timestamp']
            idx = [weather_train_df.columns.get_loc(col) for col in WEATHER_COLS_TO_STANDARIZE]
            #Prepare weather data
            weather_train_df = weather_train_df.to_numpy()
            weather_train_df, weather_train_mean, weather_train_std = standarize(weather_train_df, idx=idx, return_mean_and_std=True)
            if not erase_nans:
                weather_train_df = transform_nans(data=weather_train_df, operation='median')
            if not charge_test:
                self.weather_train_df = {tuple(key) : value for key, value in zip(weather_train_df[:, WEATHER_KEYS_IDX].astype(np.int), weather_train_df[:, WEATHER_VALUES_IDX])}
                #Prepare df data
                self.train_df = train_df.to_numpy().astype(np.float32)
                if not erase_nans:
                    self.train_df = transform_nans(data=self.train_df, operation='median')

                self.train_charged = True
            else:
                del weather_train_df
                del train_df
                self.train_charged = False
        if charge_test:
            test_df = pd.read_csv(os.path.join(root, 'test.csv'))
            # We save the time as day and hour in separated fields, because hour and season are better descriptors
            test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
            test_df['day_year'] = test_df["timestamp"].dt.dayofyear
            test_df['month'] = test_df["timestamp"].dt.month
            test_df['day'] = test_df["timestamp"].dt.dayofweek
            test_df['hour'] = test_df["timestamp"].dt.hour
            del test_df['timestamp']
            # We save the time as day and hour in separated fields, because hour and season are better descriptors
            weather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))
            weather_test_df["timestamp"] = pd.to_datetime(weather_test_df["timestamp"])
            # We save the time as day and hour in separated fields, because hour and season are better descriptors
            weather_test_df['day_year'] = weather_test_df["timestamp"].dt.dayofyear
            weather_test_df['month'] = weather_test_df["timestamp"].dt.month
            weather_test_df['day'] = weather_test_df["timestamp"].dt.dayofweek
            weather_test_df['hour'] = weather_test_df["timestamp"].dt.hour
            del weather_test_df['timestamp']
            # Prepare weather data
            idx = [weather_test_df.columns.get_loc(col) for col in WEATHER_COLS_TO_STANDARIZE]
            weather_test_df = weather_test_df.to_numpy()
            weather_test_df = standarize(weather_test_df, idx=idx, mean=weather_train_mean, std=weather_train_std)
            if not erase_nans:
                weather_test_df = transform_nans(data=weather_test_df, operation='median')
            self.weather_test_df = {tuple(key) : value for key, value in zip(weather_test_df[:, WEATHER_KEYS_IDX].astype(np.int), weather_test_df[:, WEATHER_VALUES_IDX])}
            # Prepare df data
            self.test_df = test_df.to_numpy().astype(np.float32)
            if not erase_nans:
                self.test_df = transform_nans(data=self.test_df,operation='median')

            self.test_charged = True

        #Change Train or Test for charge one dataset or another
        self.charge = 'Train'

        to_print = 'Charged'
        if charge_train:
            to_print += ' Train Set '
        if charge_test:
            to_print += ' Test Set '
        print(to_print)

    def __len__(self):
        return len(self.train_df) if self.charge.lower() == 'train' else len(self.test_df)


    def __getitem__(self, idx):
        building_meta = self.building_meta_df
        if self.charge.lower() == 'train':
            weather = self.weather_train_df
            df_data = self.train_df[idx]
        else:
            weather = self.weather_test_df
            df_data = self.test_df[idx]
            #Reestructured in the same way than the training data
            df_data[0], df_data[1], df_data[2] = df_data[1], df_data[2], df_data[0]

        building_data = building_meta[int(df_data[0])]
        weather_keys = (int(building_data[0]), int(df_data[3]), int(df_data[-1]))
        if weather_keys not in weather or np.isnan(weather[weather_keys]).any():
            weather_keys = get_nearest_hour_info(weather_keys, weather)
        weather_data = weather[weather_keys]
        x = construct_x(df_data, building_data, weather_data)
        #In case of test Y will mean the id for the CSV
        y = df_data[Y_IDX]
        if np.sum(np.isnan(x))>0:
            x = np.zeros(shape=INPUT_LEN, dtype=np.float32)
            y = np.float32(0.)
        return (x, y)


def construct_x(df_data, building_data, weather_data):
    x = np.zeros(shape=INPUT_LEN, dtype=np.float32)
    #One Hot Encoding the Meter, Month, Day of Week and Hour Variables
    x[int(df_data[METER_IDX])] = 1.
    x[METER_LEN+int(df_data[DF_MONTH_IDX])] = 1.
    x[METER_LEN+MONTHS+int(df_data[DF_DAYWEEK_IDX])] = 1.
    x[METER_LEN + MONTHS + DAYS + int(df_data[DF_HOUR_IDX])] = 1.
    #One Hot Encoding the Primary Use Variable
    x[DF_DATA_LEN+int(building_data[PRIMARY_USE_IDX])] = 1.
    #Including Square Feet, Year Built and Floor Count
    x[DF_DATA_LEN+TYPES_LEN:DF_DATA_LEN+BUILDING_DATA_LEN] = building_data[-3]
    #Including Air_temperature, Cloud_Coverage, Dew_temperature, Precip Depth 1 hour,
    #Sea Level Presure, Wind Direction and Wind Speed
    x[DF_DATA_LEN+BUILDING_DATA_LEN:] = weather_data
    return x

def standarize(data, idx, mean=None, std=None, return_mean_and_std=False):
    if mean is None:
        mean = np.nanmean(data[...,idx],axis=0)
    if std is None:
        std = np.nanstd(data[...,idx],axis=0)
    data[...,idx] = (data[...,idx]-mean)/std
    if not return_mean_and_std:
        return data
    else:
        return data, mean, std

def transform_nans(data, operation='mean'):
    for i in range(data.shape[-1]):
        if operation.lower() == 'mean':
            data[np.isnan(data[...,i]),i] = np.nanmean(data[...,i])
        elif operation.lower() == 'median':
            data[np.isnan(data[..., i]), i] = np.nanmedian(data[..., i])
        elif operation.lower() == '-1':
            data[np.isnan(data[..., i]), i] = -1
    return data

def get_nearest_hour_info(key, data):
    site_id, day, hour = key
    #Search for rows without nans
    while(not (site_id, day, hour) in data):
        hour = (hour+1)%24
        day += int(hour == 0)

    return (site_id, day, hour)