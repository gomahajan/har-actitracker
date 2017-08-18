import pandas as pd
import numpy as np
from scipy import stats

def read_data(file_path, usage):
    '''
    @param file_path: Path for file containing data
    @param usage: Amount of data to be used
    @return: DataFrame
    '''
    column_names = ['user-id','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path,header = None, names = column_names, comment=';')
    data = data.dropna(axis=0, how='any')
    data = data[1:(usage*data.shape[0])//100]
    return data


def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size // 2)


def create_segments(data,window_size = 90):
    segments = np.empty((0,window_size,3))
    labels = np.empty((0))
    for (start, end) in windows(data['timestamp'], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if(len(data['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x,y,z])])
            labels = np.append(labels,stats.mode(data["activity"][start:end])[0][0])
    return segments, labels