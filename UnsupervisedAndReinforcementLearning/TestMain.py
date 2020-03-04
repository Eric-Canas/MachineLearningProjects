import numpy as np
from LargeDatasetKmeans import RecursivePartitionKmeans
from sklearn.cluster import KMeans
import os
import time
from matplotlib import pyplot as plt

clusters = range(4,129,4)
repetitions = 3
algorithms_to_use = (('RPKM', RecursivePartitionKmeans), ('Kmeans++', KMeans),
                                                        ('Kmeans (Random Init)', KMeans))

def plot(path, array, title, x_axis, legend, xlabel, ylabel):
    if len(array.shape) == 2:
        for row in array:
            plt.plot(x_axis,row)
    else:
        plt.plot(x_axis,array)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend)

    plt.savefig(path)
    plt.close()

def run(algorithm, name, n_clusters):
    if 'Kmeans' in name:
        if '++' in name:
            alg = algorithm(n_clusters = n_clusters, init='k-means++', n_init=1, precompute_distances = False,
                            algorithm = 'full')
        else:
            alg = algorithm(n_clusters=n_clusters, init='random', n_init=1, precompute_distances = False,
                            algorithm = 'full')
    else:
        alg = algorithm(n_clusters=n_clusters, verbose=True)
    start = time.time()
    fitted_algorithm = alg.fit(X=X)
    end_time = time.time() - start
    if 'Kmeans' in name:
        operations = fitted_algorithm.n_iter_*len(X)*n_clusters
        inertia = fitted_algorithm.inertia_
        iterations = fitted_algorithm.n_iter_

    else:
        operations = fitted_algorithm.distance_operations
        inertia = fitted_algorithm.get_inertia(X=X, labels=fitted_algorithm.labels)
        iterations = fitted_algorithm.n_iter
    return end_time, operations, inertia, iterations


def plot_averages(f, algorithm_time, operations_done, inertias_done, iterations_done):
    f.write('\t\t\tTime: ' + str(algorithm_time) + 's\n')
    print('\t\t\tTime: ' + str(algorithm_time) + 's')
    f.write('\t\t\tOperations: ' + str(operations_done)+'\n')
    print('\t\t\tOperations: ' + str(operations_done))
    f.write('\t\t\tInertia: ' + str(inertias_done)+'\n')
    print('\t\t\tInertia: ' + str(inertias_done))
    f.write('\t\t\tIterations: ' + str(iterations_done)+'\n\n')
    print('\t\t\tIterations: ' + str(iterations_done))

def get_string_as_csv(array, delimiter=',', round=1):
    txt = ''
    for row in array:
        for value in row:
            txt += str(np.round(value,decimals=round))+delimiter
        txt = txt[:-len(delimiter)]+'\n'
    return txt

if __name__ == '__main__':
    files = os.listdir('SampleDatasets')
    sizes = [os.path.getsize(os.path.join('SampleDatasets',file)) for file in files]
    files = np.array(files)[np.argsort(sizes)]
    for dataset in files:
        if '#' in dataset:
            continue
        X = np.load(os.path.join('SampleDatasets', dataset))
        dataset_name = dataset[:-len('.npy')]+'('+str(X.shape[0])+'x'+str(X.shape[1])+')'
        root = os.path.join('experiments', dataset_name)
        if not os.path.exists(root):
            os.makedirs(root)
        with open(os.path.join(root,'results.txt'), mode='w') as f:
            f.write('DATASET: ' + dataset+'\n')
            print('DATASET: ' + dataset)
            time_array = []
            operations = []
            inertias = []
            iterations = []
            for name, algorithm in algorithms_to_use:
                f.write('\t'+name+':\n')
                print('\t'+name+':')
                time_array.append([])
                operations.append([])
                inertias.append([])
                iterations.append([])
                for n_clusters in clusters:
                    f.write('\t\tClusters: '+str(n_clusters)+'\n')
                    print('\t\tClusters: '+str(n_clusters))
                    times = []
                    operations_iter = []
                    inertias_iter = []
                    iterations_iter = []
                    for repeat in range(repetitions):
                        execution_time, operations_experiment, inertia_given, iterations_given =\
                            run(algorithm=algorithm, name=name, n_clusters=n_clusters)
                        times.append(execution_time)
                        operations_iter.append(operations_experiment)
                        inertias_iter.append(inertia_given)
                        iterations_iter.append(iterations_given)
                    algorithm_time = np.round(np.mean(times), 5)
                    operations_done = np.round(np.mean(operations_iter), 1)
                    inertias_done = np.round(np.mean(inertias_iter), 1)
                    iterations_done = np.round(np.mean(iterations_iter), 1)
                    time_array[-1].append(algorithm_time)
                    operations[-1].append(operations_done)
                    inertias[-1].append(inertias_done)
                    iterations[-1].append(iterations_done)
                    plot_averages(f, algorithm_time, operations_done, inertias_done, iterations_done)

            #--------------------------------------- PLOTS --------------------------------------------------

            plot(path=os.path.join(root, 'TimeComparisons.pdf'), array=np.array(time_array),
                 title='Mean time by cluster size for '+dataset_name.title()+' Dataset',
                 x_axis=clusters,legend=[name for name,_ in algorithms_to_use], xlabel='Clusters', ylabel='Time')
            f.write('\tTime:\n'+get_string_as_csv(time_array)+'\n')

            plot(path=os.path.join(root, 'DistanceOperationsComparisons.pdf'), array=np.array(operations),
                 title='Mean Distance Operations by cluster size for '+dataset_name.title()+' Dataset',
                 x_axis=clusters, legend=[name for name,_ in algorithms_to_use], xlabel='Clusters', ylabel='Operations')
            f.write('\tOperations:\n' + get_string_as_csv(operations)+'\n')

            plot(path=os.path.join(root, 'InertiaComparison.pdf'), array=np.array(inertias),
                 title='Inertia by cluster size for ' + dataset_name.title() + ' Dataset',
                 x_axis=clusters, legend=[name for name, _ in algorithms_to_use], xlabel='Clusters',
                 ylabel='Inertia')
            f.write('\tInertias:\n' + get_string_as_csv(inertias)+'\n')

            plot(path=os.path.join(root, 'IterationsComparison.pdf'), array=np.array(iterations),
                 title='Iterations by cluster size for ' + dataset_name.title() + ' Dataset',
                 x_axis=clusters, legend=[name for name, _ in algorithms_to_use], xlabel='Clusters',
                 ylabel='Iterations')
            f.write('\tIterations:\n' + get_string_as_csv(iterations)+'\n')