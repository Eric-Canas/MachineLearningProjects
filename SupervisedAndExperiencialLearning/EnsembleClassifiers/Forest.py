import numpy as np
from C45 import C45
from Reader import bootstrap_data
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from collections import Counter

try:
    DEFAULT_THREADS = cpu_count()
except:
    DEFAULT_THREADS = 4

LABEL, COUNTS = 0, 1
FOREST_TYPES = ['random_forest', 'decision_forest']

class Forest:
    """
    Implementation of the Random Forest using C4.5 as base classifier
    """
    def __init__(self, F, NT, threads=DEFAULT_THREADS, forest_type='random_forest'):
        """"
        :param F: Total amount of variables that each classifier will take.
        Usually much smaller than M (Total number of variables of the set),
        Usual values: sqrt(M)or int(log2(M)+1)
        :param NT: Number of trees to generate
        :param threads: Default 1. If grater than 1 the computation will be parallelized
        among that amount of threads.
        """

        self.NT = NT
        self.F = F
        self.threads = threads
        self.forest = []
        if forest_type not in FOREST_TYPES:
            raise ValueError(forest_type+' is not a valid forest type. '
                                         'Valid forest types: '+str(FOREST_TYPES))
        else:
            self.forest_type = forest_type

    def fit(self, X, Y, X_names):
        """
        Fits a C4.5 based Random Forest/Decision Forest
        :param X: 2 dimensional ndarray of inputs
        :param Y: 1 dimensional ndarray of expected outputs for each input row in X
        :return:
            Self. The trained object.
        """

        # ---------------------- PREPARING THE DATASET ---------------------------
        if self.forest_type == 'random_forest':
            # Each dataset will take only the i bootstrapped set
            def tree_params(i, X, Y):
                x, y = bootstrap_data(X_data=X, Y_data=Y)
                return {'X':x, 'Y': y, 'X_names' : X_names, 'F':self.F}

        else:
            # Select the attributes which each tree will take
            attributes = [np.random.choice(range(X.shape[-1]), self.F(), replace=False) for _ in range(self.NT)]
            # In Decision forest each tree will take the full dataset with different attributes
            tree_params = lambda i, X, Y: {'X':X[...,attributes[i]], 'Y': Y, 'X_names' : X_names[attributes[i]], 'F':None}

        # ----------------------- COMPUTING THE FOREST ---------------------------
        # If threads > 1 compute the forest parallel.
        if self.threads > 1:
            with ThreadPool(processes=self.threads) as pool:  # use all cpu cores
                # Makes the computations asynchronous
                for i in range(self.NT // self.threads):
                    async_results = [pool.apply_async(C45().fit, tree_params(j, X,Y).values())
                                     for j in range(i*self.threads,(i+1)*self.threads)]

                    self.forest.extend([async_result.get() for async_result in async_results])
                # Closses the multiprocessing pool
                pool.close(), pool.terminate(), pool.join()

                # Computes the last ones which are not multiple of threads
                self.forest.extend([C45().fit(**tree_params(last_trees, X,Y))
                                    for last_trees in range(len(self.forest), len(self.forest) + (self.NT % self.threads))])

        # Else compute it sequentially in a single core
        else:
            self.forest = [C45().fit(**(tree_params(i, X,Y)))for i in range(self.NT)]

        return self


    def predict(self, X, X_names):
        """
        Makes the prediction by voting system for all instances in X
        :param X: 2-d ndarray. Instances to predict
        :param X_names: 1-d ndarray. Names of each attribute in the array X.
        :return:
            Predictions for all instances in X
        """
        # For each instance, take the votes of the full forest
        voting = np.array([tree.predict(X=X, names = X_names) for tree in self.forest]).T
        # Makes a maximum voting of each prediction for deciding the final prediction
        predictions = [Counter(votes).most_common(1)[0][LABEL] for votes in voting]

        return predictions

    def attributes_relevance(self):
        """
        Get the relevance of each attribute, induced from the generated forest
        :return:
        """
        # Concatenate the attributes used by each tree of the forest
        attributes = np.concatenate([tree.get_attributes() for tree in self.forest])
        # Compute its full frequency
        labels, freq = np.unique(attributes, return_counts=True)
        # Order them descendant
        order = np.argsort(-freq)
        relevance, freq = labels[order], freq[order]
        # Return labels and its frequency (frequency normalized)
        return relevance, freq/np.sum(freq)

    def save_attributes_relevance_plot(self, file, title='Feature Relevance'):
        relevance, freq = self.attributes_relevance()
        from matplotlib import pyplot as plt
        relevance = [(label[:20]+'...' if len(label)>20 else label) for label in relevance]
        plt.bar(range(len(freq)), height=freq, tick_label = relevance)
        plt.title(title)
        plt.xlabel('Attribute')
        plt.ylabel('Relevance')
        if len(relevance) > 6:
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(file)
        plt.close()

