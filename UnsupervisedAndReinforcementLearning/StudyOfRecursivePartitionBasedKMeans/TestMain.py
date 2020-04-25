import numpy as np
from LargeDatasetKmeans import RecursivePartitionKmeans
from sklearn.cluster import KMeans, MiniBatchKMeans
import os
import time
from matplotlib import pyplot as plt

TIME, OPS, INERTIA = 0,1,2
clusters = range(4,129,4)
repetitions = 3
algorithms_to_use = (('RPKM', RecursivePartitionKmeans), ('Kmeans++', KMeans),
                                                        ('MinibatchKmeans', MiniBatchKMeans))
minibatches = (100,500,1000)
max_iters = range(1,7)
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


def get_inertia(X, cluster_centers, labels):
    return np.sum([np.sum(np.linalg.norm(X[labels == i]-centroid)) for i, centroid in enumerate(cluster_centers)])

def run(algorithm, name, n_clusters, X, param = None, clusters = None):
    if 'Kmeans' in name:
        if '++' in name:
            alg = algorithm(n_clusters = n_clusters, init='k-means++' if clusters is None else clusters,
                            n_init=1, precompute_distances = False,
                            algorithm = 'full')
        else:
            alg = algorithm(n_clusters=n_clusters, init='k-means++', n_init=1, batch_size=param)
    else:
        alg = algorithm(n_clusters=n_clusters, max_iter=param)
    start = time.time()
    fitted_algorithm = alg.fit(X=X)
    end_time = time.time() - start
    if 'Kmeans' in name:
        if '++' in name:
            operations = fitted_algorithm.n_iter_*len(X)*n_clusters
        else:
            operations = fitted_algorithm.n_iter_ * len(X) * n_clusters/param
        inertia = get_inertia(X=X, cluster_centers=fitted_algorithm.cluster_centers_, labels=fitted_algorithm.labels_)
        iterations = fitted_algorithm.n_iter_

    else:
        operations = fitted_algorithm.distance_operations
        inertia = get_inertia(X=X, cluster_centers=fitted_algorithm.cluster_centers, labels=fitted_algorithm.labels)
        iterations = fitted_algorithm.n_iter
    return end_time, operations, inertia


def quality(n_clusters, X, max_iters):
    rpkms = RecursivePartitionKmeans(n_clusters=n_clusters, max_iter=max_iters).fit(X)
    error_rpkms = get_inertia(X=X, cluster_centers=rpkms.cluster_centers, labels=rpkms.labels)
    kmeans = KMeans(n_clusters=n_clusters, init=rpkms.cluster_centers, n_init=1, precompute_distances=True, algorithm='full').fit(X)
    error_kmeans = get_inertia(X=X, cluster_centers=kmeans.cluster_centers_, labels=kmeans.labels_)
    return (error_kmeans-error_rpkms)/error_kmeans


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

def simple_tests():
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
                        execution_time, operations_experiment, inertia_given =\
                            run(algorithm=algorithm, name=name, n_clusters=n_clusters, X=X, param=6)
                        times.append(execution_time)
                        operations_iter.append(operations_experiment)
                        inertias_iter.append(inertia_given)
                        iterations_iter.append(0)
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

def get_artificial_dataset(N, d, k):
    #select a different but similar sigma for each gaussian
    sigma = np.random.uniform(low=1.6,high=2.6,size=k)
    #Each one will have an overlapping of no more than 2sigma (0.021 of probability at each gaussian queue)
    locations = np.cumsum(sigma*(2+np.random.uniform(low=0,high=0.3,size=k)))

    nk = N//k
    X = np.empty(shape=(nk*k, d),dtype=np.float32)
    for i in range(k):
        X[i*nk:(i+1)*nk] = np.random.normal(loc=locations[i], scale=sigma[i], size=(nk,d))
    return X

def get_real_dataset(N, d):
    dataset = np.load(os.path.join('SampleDatasets', 'RealDataset.npy'))
    dataset = dataset[...,np.random.choice(range(dataset.shape[-1]),size=d, replace=False)]
    dataset = dataset[np.random.choice(range(len(dataset)), size=N, replace=False)]
    return dataset
def plot_complete(Kmeanspp_results, minibatches_results, RPKM_results, legend, n, d, k, ax, result_place=1):
    print("One Plot Completed")
    RPKM_results = np.array(RPKM_results).swapaxes(0, 1)

    for marker in ('-', 'o'):
        ax.plot(n, np.array(Kmeanspp_results)[:,result_place], marker, color='C'+str(0))
        [ax.plot(n, exp, marker, color='C'+str(i+1))
         for i, exp in enumerate(np.array(minibatches_results)[...,result_place].swapaxes(0,1))]
        [ax.plot(n, exp, marker, color='C'+str(i+len(minibatches)+1))
         for i, exp in enumerate(np.array(RPKM_results)[..., result_place])]
    #ax.set_xscale('log')
    #ax.set_yscale('log')

def plot_quality(error, ax):
    print("One Plot Completed")
    for marker in ('-', 'o'):
        [ax.plot(max_iters, row, marker, color='C' + str(i))
         for i, row in enumerate(error)]

def time_experimentation():
    N = [1000, 1770, 3160, 5620, 10000, 17700, 31600, 56200, 100000, 177000, 316000, 562000, 1000000, 1770000, 3160000]
    d = [2,4,8]
    K = [3,9]
    TIME, OPS, INERTIA = 0,1,2
    executions = 5
    fig, ax = plt.subplots(len(d), len(K), sharex='col', sharey='row')
    legend = ['Kmeans++']+['MiniBatch: '+str(m) for m in minibatches]+ ['RPKM: '+str(i) for i in max_iters]

    fig.suptitle("Total time spent for Artificial Datasets")
    for i, d_ in enumerate(d):
        for j, k_ in enumerate(K):
            Kmeanspp_results, minibatches_results, RPKM_results = [], [], []
            for n_ in N:
                X = lambda: get_artificial_dataset(N=n_, d=d_, k=k_)#get_artificial_dataset(N=n_, d=d_, k=k_)
                Kmeanspp_results.append(np.median([run(algorithm=KMeans, name="Kmeans++", n_clusters=k_, X=X())for i in range(executions)],axis=0))
                print("Kmeans "+str(n_))
                minibatches_results.append([np.median([run(algorithm=MiniBatchKMeans, name="MinibatchKmeans", n_clusters=k_,
                                                        X=X(), param=minibatch) for i in range(executions)],axis=0)
                                       for minibatch in minibatches])
                print("MiniBatch "+str(n_))
                RPKM_results.append([np.median([run(algorithm=RecursivePartitionKmeans, name="RPKM", n_clusters=k_, X=X(),
                                                  param=m) for i in range(executions)],axis=0)
                                       for m in max_iters])
                print("RPKM "+str(n_))
            plot_complete(Kmeanspp_results=Kmeanspp_results, minibatches_results=minibatches_results, RPKM_results=RPKM_results,
                          legend=legend, n=N, d=d_, k=k_, result_place=TIME, ax=ax[i,j])

    [ax[0,j].title.set_text('K='+str(K[j])) for j in range(len(K))]
    [(ax[i, -1].yaxis.set_label_position("right"), ax[i, -1].set_ylabel('d=' + str(d[i]))) for i in range(len(d))]
    ax[len(d)//2, 0].set_ylabel("Operations Computed")
    ax[-1, 0].set_xlabel("Dataset Size")
    fig.legend(ax[0,0].get_lines()[0:len(legend)], legend)
    plt.show()

def quality_experimentation():
    N = [1000, 10000, 100000, 1000000, 3160000]
    d = [2,4,8]
    K = [3,9]
    max_iters = range(1,7)
    executions = 5
    fig, ax = plt.subplots(len(d), len(K), sharex='col', sharey='row')
    legend = ['Data Size: '+str(i) for i in N]

    fig.suptitle("Standard Error over Real Datasets")
    for i, d_ in enumerate(d):
        for j, k_ in enumerate(K):
            error = []
            for n_ in N:
                error.append(np.mean([[quality(n_clusters=k_, X=get_real_dataset(N=n_, d=d_), max_iters=m) for m in max_iters] for i in range(executions)], axis=0))
            plot_quality(error, ax=ax[i,j])

    [ax[0,j].title.set_text('K='+str(K[j])) for j in range(len(K))]
    [(ax[i, -1].yaxis.set_label_position("right"), ax[i, -1].set_ylabel('d=' + str(d[i]))) for i in range(len(d))]
    ax[len(d)//2, 0].set_ylabel("Standard Error")
    ax[-1, 0].set_xlabel("RPKM Iteration")
    fig.legend(ax[0,0].get_lines()[0:len(legend)], legend)
    plt.show()

if __name__ == '__main__':
    time_experimentation()