import numpy as np
from LargeDatasetKmeans import RecursivePartitionKmeans

X = np.load('SampleDatasets/grid.npy')
RecursivePartitionKmeans().fit(X=X)
