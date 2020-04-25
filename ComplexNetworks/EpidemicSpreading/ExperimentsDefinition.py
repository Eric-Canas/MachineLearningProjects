from ERModelGenerator import getERModel_Np
from CMModelGenerator import getCMModel
import numpy as np
import networkx as nx
import os

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

experiments = {'Erdós-Rényi with N = 1000 - p = 0.05': getERModel_Np(N=1000, p=0.05),
               'Scale Free γ = 2.5 - N = 5000': getCMModel(N=5000, P_k=power_law_dist(N=5000, gamma=2.5)),
               'Real Network - Dolphins': nx.read_pajek(os.path.join('RealNetworks', 'dolphins.net'))}
