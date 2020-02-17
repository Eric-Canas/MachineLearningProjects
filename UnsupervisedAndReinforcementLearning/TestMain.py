import numpy as np
from LargeDatasetKmeans import RecursivePartitionKmeans

X = np.load('SampleDatasets/grid.npy')
G = RecursivePartitionKmeans(verbose=True).fit_predict(X=X)
print("H")
