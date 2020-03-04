import numpy as np
import os
if __name__ == '__main__':
    for dataset in os.listdir('SampleDatasets'):
        if '.npy' not in dataset:
            data = np.genfromtxt(os.path.join('SampleDatasets', dataset),delimiter=',')[:,:-1]
            nans = np.any(np.isnan(data), axis=0)
            data = data[:, nans == False]
            np.save(file=os.path.join('SampleDatasets',dataset[:-len('.data')]), arr=data)