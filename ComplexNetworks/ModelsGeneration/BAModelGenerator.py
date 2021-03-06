import networkx as nx
import numpy as np

# Constants
NODE, DEGREE = 0, 1


def getBAModel(N, m_0, m):
    """
    Generates a Barabási-Albert model with N nodes, m_0 initial nodes and m edges for each new node.
    Type of model: according to a degree distribution.

    :param N: int. Number of nodes of the network.
    :param m_0: int. Number of initial nodes of the network (0 <= m_0 <= N).
    :param m: int. Number of edges for each new node (0 <= m <= m_0).

    :return: a networkx network generated by the Barabási-Albert model.
    """

    # STEP 1: Check if parameters are valid
    if not 0 <= m_0 <= N:  # Initial number of nodes
        raise ValueError("The number of initial nodes (m_0) of a Barabási-Albert model with " + str(N) + " nodes must "
                        "be in the range [0, " + str(N) + "]. Given value: " + str(m_0))

    elif not 0 <= m <= m_0:  # Number of edges for each new node
        raise ValueError("The number of edges (m) of a Barabási-Albert model with " + str(m_0) + " initial nodes must "
                         "be in the range [0, " + str(m_0) + "]. Given value: " + str(m))

    # STEP 2: Generate the network
    graph = nx.complete_graph(n=m_0)  # Create a fully-connected graph (all possible connections) with m_0 nodes

    # Iterate until the graph has N nodes (starting from m_0)
    for u in range(m_0, N):
        # Compute the probability to connect to each node (probability) using the preferential attachment formula,
        # that uses the degree of each node (actual_degrees) [Formula (node i): degree(i)/sum of all degrees)]
        actual_degrees = np.array(graph.degree)
        nodes_id, probability = actual_degrees[:, NODE], actual_degrees[:, DEGREE]/np.sum(actual_degrees[:, DEGREE])
        # Represent each probability as the cumulative sum of the previous probabilities so that they are
        # distributed in the range [0, 1] proportionally and can be interpreted as bins
        cum_probability = np.cumsum(probability)
        graph.add_node(node_for_adding=u)  # Add a new node, u

        # Iterate over each edge of the new node u
        for edge in range(m):
            # Compute to which node v connect the node u using a random uniform probability.
            # This is done selecting the node whose range includes the value obtained by np.random.uniform()
            v = nodes_id[np.argmax(cum_probability >= np.random.uniform())]
            # If this connection existed previously, repeat the previous line until a valid edge is found
            while graph.has_edge(u=u, v=v):
                v = nodes_id[np.argmax(cum_probability >= np.random.uniform())]

            graph.add_edge(u_of_edge=u, v_of_edge=v)  # Add the edge to the graph

    return graph
