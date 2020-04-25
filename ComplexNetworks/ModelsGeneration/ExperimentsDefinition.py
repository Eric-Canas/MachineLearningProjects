import networkx as nx
import numpy as np

from decimal import Decimal
from math import factorial

from ERModelGenerator import getERModel_NK, getERModel_Np
from CMModelGenerator import getCMModel
from BAModelGenerator import getBAModel
from WSModelGenerator import getWSModel


def poisson_dist_ER_based(N, K):
    """
    Generates a Poisson distribution based on the degree distribution obtained by a ER G(N,K) model.

    :param N: int. Number of nodes of the network.
    :param K: int. Number of edges of the network (0 <= K <= N(N-1)/2).

    :return: ndarray of shape (n_degrees, 2) (*) with the degree distribution of the network.
    (*) It must have the form [[degree_0, p_0], [degree_1, p_1], ..., [degree_n, p_n]] and all the p_i must add 1,
    since they are probabilities.
    """

    # STEP 1: Generate the ER G(N,K) model
    model = getERModel_NK(N=N, K=K)

    # STEP 2: Obtain a Poisson distribution from the degree distribution of the model
    degrees = np.array([d for _, d in model.degree()])  # Get the degree of each node
    histogram, bins = np.histogram(degrees, bins=range(N))  # Compute the histogram given the degree sequence 'degrees'
    # Transform the histogram into a probability distribution (so that it adds 1)
    histogram = histogram / np.sum(histogram)

    # Return the histogram as a probability distribution with the shape [[degree_0, p_0], ..., [degree_n, p_n]]
    return np.concatenate((bins[1:], histogram)).reshape(2, -1).T


def poisson_dist(N, lam):
    """
    Generates a Poisson distribution with lambda 'lam'.

    :param N: int. Number of nodes of the network to which the obtained distribution will be applied. Used to estimate
    a representative number of samples of the distribution.
    :param lam: float. Average degree of the network.

    :return: ndarray of shape (n_degrees, 2) (*) with the degree distribution of the network.
    (*) It must have the form [[degree_0, p_0], [degree_1, p_1], ..., [degree_n, p_n]] and all the p_i must add 1,
    since they are probabilities.
    """

    # Get a number of N*100 random samples using a Poisson distribution with lambda=lam
    samples = np.random.poisson(lam, size=N*100)
    # Compute the histogram given the samples obtained before
    histogram, bins = np.histogram(samples, bins=range(N))
    # Transform the histogram into a probability distribution (so that it adds 1)
    histogram = histogram / np.sum(histogram)

    # Return the histogram as a probability distribution with the shape [[degree_0, p_0], ..., [degree_n, p_n]]
    return np.concatenate((bins[1:], histogram)).reshape(2, -1).T

def ws_dist(N, K, p):
    """
    Generates a Watts-Strogatz distribution for 'N' nodes, 'K' mean degree and probability 'p'.

    :param N: int. Number of nodes of the network.
    :param K: int. Mean degree of the network (even, 0 <= k < N).
    :param p: float. Rewiring probability (0. <= p <= 1.).

    :return: ndarray of shape (n_degrees, 2) (*) with the degree distribution of the network.
    (*) It must have the form [[degree_0, p_0], [degree_1, p_1], ..., [degree_n, p_n]] and all the p_i must add 1,
    since they are probabilities.
    """

    k_range = range(N)  # Generate all possible degrees (from 0 to N-1)

    # Shortcuts and common functions used in the Watts-Strogatz distribution
    k_2 = K//2  # Shortcut of the common operation K/2
    fK = lambda k, K: min(k-k_2, k_2)  # f(k,K) function is defined as the lambda 'fk(k,K)'
    # Factorial of n over k is defined as the lambda 'combinatorial(n,k)'
    combinatorial = lambda n, k: Decimal(factorial(n)/(factorial(k)*factorial(n-k)))

    distribution = []  # List to store the results of the degree distribution

    for k in k_range:  # Iterate over all the possible degrees
        # Obtain the probability for each degree following the formula
        probability = [combinatorial(k_2, n)*Decimal((1-p)**n)*Decimal(p**(k_2-n))*
                       ((Decimal((p*k_2))**Decimal(k-k_2-n))/Decimal(factorial(k-k_2-n)))*
                       Decimal(np.e**(-p*k_2)) for n in range(fK(k, K) + 1)]

        distribution.append((k, np.sum(probability)))  # Append the result of the degree k to 'distribution'

    return np.array(distribution)


def power_law_dist(N, gamma):
    """
    Generates a Power Law distribution with gamma 'gamma'.

    :param N: int. Number of nodes of the network to which the obtained distribution will be applied. Used to estimate
    a representative number of samples of the distribution.
    :param gamma: float. Negative exponent of the distribution.

    :return: ndarray of shape (n_degrees, 2) (*) with the degree distribution of the network.
    (*) It must have the form [[degree_0, p_0], [degree_1, p_1], ..., [degree_n, p_n]] and all the p_i must add 1,
    since they are probabilities.
    """

    # Create an array with all the possible degrees of a network with N nodes (starting by 5 instead by 0 so that the
    # gamma estimation is more accurate), this is: [5, N-1]
    degrees = np.arange(5, N, dtype=np.float)
    distribution = degrees**-gamma  # Compute the distribution using the Power Law formula
    # Transform 'distribution' into a probability distribution (so that it adds 1)
    distribution = distribution/np.sum(distribution)

    # Return the distribution as a probability distribution with the shape [[degree_0, p_0], ..., [degree_n, p_n]]
    return np.concatenate((degrees, distribution)).reshape(2, -1).T


# Define the complete set of experiments for this exercise (in a dictionary form)
# Each dictionary entry represents each one of the models tested and inside, another dictionary with the following keys:
# * function: Python function implemented to generate the model
# * parameters: list of dictionaries with the parameters used for each experiment
# * preferred layout: layout of the nodes of the network
experiments = {
                #'Erdós-Rényi N-p':
                   #{'function': getERModel_Np,
                    #'parameters': [{'N': 50, 'p': 0.1}, {'N': 50, 'p': 0.75}, {'N': 100, 'p': .5},
                                  # {'N': 1000, 'p': .25}, {'N': 10000, 'p': 0.01}],
                    #'preferred layout': nx.kamada_kawai_layout
                    #},
                #'Watts-Strogatz':
                   #{'function': getWSModel,
                    #'parameters': [{'N': 50, 'k': 2, 'p': 0.1}, {'N': 50, 'k': 2, 'p': 1.},
                                   #{'N': 100, 'k': 4, 'p': .75}, {'N': 1000, 'k': 50, 'p': .5},
                                   #{'N': 1000, 'k': 50, 'p': 0.}, {'N': 10000, 'k': 100, 'p': .25}],
                    #'preferred layout': nx.circular_layout
                   #},

                'Barabási-Albert':
                    {'function': getBAModel,
                     'parameters': [  #
                            {'N': 1000, 'm_0': 25, 'm': 25},
                          {'N': 1000, 'm_0': 50, 'm': 50},
                         {'N': 1000, 'm_0': 50, 'm': 25}],
                     'preferred layout': nx.kamada_kawai_layout
                     }
            }


def getAllExperiments():
    """
    Obtains a generator for each one of the experiments defined in the dictionary 'experiments'

    :return: generator formed by four elements (model_name, experiment['function'],
    parameters, experiment['preferred layout'])
    """

    # Iterate over each pair key, value (model_name, experiments) of the dictionary 'experiments'
    for model_name, experiment in experiments.items():
        # Iterate over each experiment defined in the current model
        for parameters in experiment['parameters']:
            # Return the generator with all the elements needed for executing the experiment
            yield model_name, experiment['function'], parameters, experiment['preferred layout']
