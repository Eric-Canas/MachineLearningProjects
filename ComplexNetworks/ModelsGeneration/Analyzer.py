import networkx as nx
import numpy as np
import warnings
import os

from matplotlib import pyplot as plt
from textwrap import wrap

class Analyzer:
    """
    The purpose of this class is to obtain an analysis of the structural descriptors of the provided network.
    """
    def __init__(self, graph=None, root='networks', dir='toy', file=''):
        """
        If a graph in NetworkX format is provided, initialize the Analyzer object with it, else, read
        the graph from the Pajek format file 'file' from the directory 'root/dir/'.

        :param root: String. Root directory where graphs directories are located.
        :param dir: String. Graph directory within Root where network files are located.
        :param file: String. Name of the network file to open.
        """

        if graph is None:
            self.graph = nx.read_pajek(os.path.join(root, dir, file))
        else:
            self.graph = graph

    def get_numeric_descriptors(self):
        """
        Generates a dictionary with the following network descriptors:
        Number of nodes, Number of edges, Max degree, Min degree,
        Average degree, Average clustering coefficient,
        Degree assortativity, Average path length and Diameter.

        :return Dictionary with all the named network numeric descriptors.
        """

        # Define the 'descriptors' dictionary containing all the previously mentioned network descriptors.
        # Each key of the dictionary is a different descriptor obtained by a NetworkX function
        # except 'Max degree', 'Min degree' and 'Average degree', which uses a function from ours
        descriptors = {'Number of nodes': self.graph.number_of_nodes(),
                       'Number of edges': self.graph.number_of_edges(),
                       'Max degree': self.degree_descriptors(descriptor='max'),
                       'Min degree': self.degree_descriptors(descriptor='min'),
                       'Average degree': self.degree_descriptors(descriptor='mean'),
                       'Average clustering coefficient': nx.average_clustering(G=nx.Graph(self.graph)),
                       'Degree assortativity': nx.degree_assortativity_coefficient(self.graph),
                       'Average path length': nx.average_shortest_path_length(self.graph),
                       'Diameter': nx.diameter(self.graph)}

        return descriptors  # Return the descriptor dictionary

    def degree_descriptors(self, descriptor):
        """
        Gets the max, min or average degree of a network.

        :param descriptor: 'max', 'min' or 'mean'. Descriptor to extract
        :return Integer value for min or max. Float value for average.
        """

        # Read the degree of the nodes in the graph as a dtype named array (first col: node, second col: degree)
        degrees = np.array(list(self.graph.degree), dtype=[('node', '<U128'), ('degree', np.int)])
        # Get a dictionary with the numpy function which corresponds to each possible descriptor value
        func = {'max': np.max, 'min': np.min, 'mean': np.mean}
        # Apply the requested function to the 'degree' column of the array
        return func[descriptor](degrees['degree'])

    def get_node_descriptors(self):
        """
        Generates a dictionary of dictionaries with the following node descriptors:
        Degree, Strength, Clustering coefficient, Average path length,
        Maximum path length, Betweenness, Eigenvector centrality and PageRank.

        :return Dictionary of dictionaries where the first key is the node name and second key the descriptor name.
        """

        # First, obtain the descriptors where the full graph must be analyzed
        betweenness = nx.betweenness_centrality(self.graph)
        eigenvector_centrality = nx.eigenvector_centrality(nx.Graph(self.graph))
        pagerank = nx.pagerank(nx.Graph(self.graph))

        # Define the dictionary to return
        node_descriptors = {}

        # Iterate over all nodes of the graph
        for node in self.graph.nodes:

            node_descriptors[node] = {}  # Define a new dictionary within node_descriptors per node

            node_descriptors[node]['Degree'] = self.graph.degree[node]
            node_descriptors[node]['Strength'] = np.sum([weight for _, _, weight in self.graph.edges(node, 'weight')])
            node_descriptors[node]['Clustering coefficient'] = nx.clustering(nx.Graph(self.graph), node)
            node_descriptors[node]['Average path length'] = np.mean(list(nx.shortest_path_length(self.graph, node).values()))
            node_descriptors[node]['Maximum path length'] = nx.eccentricity(self.graph, v=node)
            # Get the results of the previously computed nodes for the current node
            node_descriptors[node]['Betweenness'] = betweenness[node]
            node_descriptors[node]['Eigenvector centrality'] = eigenvector_centrality[node]
            node_descriptors[node]['PageRank'] = pagerank[node]

        return node_descriptors  # Return the second-order dictionary

    def plot_distribution(self, bins=10, log_log=False, file=None, path=None, plot_histogram=None):
        """
        Plots (or saves) the Probability Degree Distribution (PDF) and the Complementary Cumulative Degree Distribution
        (CCDF) histograms in linear or log-log scale of the graph.

        :param bins: int. Number of bins to use in the histogram.
        :param log_log: bool. Apply log_log scale?
        :param file: String. Name of the file (If given, add it to the title of the plot).
        :param path: String. Location to store the plot generated by this function.
        :param plot_histogram: Histogram already computed to be plotted (for the theoretical results).
        """

        # Get the degrees of all the nodes in the graph
        degrees = np.array([d for _, d in self.graph.degree()])
        # Calculate the gamma exponent of the degree distribution (using the MLE formula)
        gamma = 1 + len(degrees)*((np.sum(np.log(degrees/(np.min(degrees)-0.5))))**-1)

        # First, check if the degree distribution of the network must be represented as a Dirac delta function
        if degrees.min() == degrees.max():
            bins = np.arange(0, 2 * degrees.max())
        else:  # Divide the bins differently depending on the scale used
            if log_log:  # 10 bins for log-log scale
                bins = np.logspace(np.log10(degrees.min()), np.log10(degrees.max()), min(bins, degrees.max()))

            else:  # Same number of bins as the maximum number of degrees for linear scale
                bins = np.arange(degrees.min(), degrees.max())

        # Calculate the histogram with the given bins
        histogram, bins = np.histogram(degrees, bins=bins)
        # Transform the histogram into a probability distribution (so that it adds 1)

        # In case the 'theoretical' histogram is also provided (excluding Dirac delta function case)
        if plot_histogram is not None and degrees.min() != degrees.max():
            # Arrange it so that it contains the same bins that the equivalent histogram obtained during the
            # experimentation and therefore, both can be comparable
            histogram = [np.sum(plot_histogram[np.bitwise_and(plot_histogram[:, 0] > bins[bin-1],
                        plot_histogram[:, 0] <= bins[bin]), 1]) for bin in range(1, len(bins))]
        # With this, 'histogram' represents now the PDF plot
        histogram = histogram/np.sum(histogram)
        try:
            # Plot the PDF and the CCDF
            # CCDF is the complementary of CDD (1-CDD). CDD is the cumulative sum of the PDF
            for title in ('Probability Degree Distribution', 'Complementary Cumulative Degree Distribution'):

                title += ". Gamma - " + str(np.round(gamma, decimals=3))  # Add the gamma value to the title

                # If we compute the CCDF, we have to modify the histogram to get the cumulative sum
                if 'Cumulative' in title:
                    histogram = 1-np.cumsum(histogram)

                # Choose the scale for the axes
                if log_log:  # Log-log scale
                    # Set the width of the bins equal for all of them
                    plt.bar(bins[1:], histogram, width=[bins[i]-bins[i-1] for i in range(1, len(bins))])
                    # Set the scale of x and y axes to log
                    plt.xscale('log')
                    plt.yscale('log')

                else:  # Linear scale
                    plt.bar(bins[1:], histogram)

                if file is not None:  # Add the name of the network to the plot title if given
                    title += ' (' + file + ')'

                # Plot titles and name of the axis
                plt.title("\n".join(wrap(title, 60)))
                plt.xlabel('Degree')
                plt.ylabel('Probability')

                # Show or save the plot
                if file is None:  # Show the plot if the title of the file is not given
                    plt.show()
                else:  # Save in 'path' otherwise
                    plt.savefig(os.path.join(path, title + '.png'))

                plt.close()

        except:
            # Catch the same-degree network exception (it would generate an histogram with only one bin)
            warnings.warn("Nonsensical distribution: all the nodes with k>0 have the same degree.")
            plt.close()

    def plot_graph(self, layout=nx.kamada_kawai_layout, alg_name='', parameters={}, save_at=None):
        """
        Plots (or saves) the network stored in self.graph.

        :param layout: Layout of the nodes of the represented network.
        :param alg_name: String. Name of the model used to generate the network.
        :param parameters: Dictionary. Parameters of the aforementioned model used to generate the network.
        :param save_at: String. Location to store the plot generated by this function.
        """

        # Define the plot of the network and its characteristics (pink nodes with 95% of opacity and without labels)
        nx.draw_networkx(self.graph, pos=layout(G=self.graph), with_labels=False, node_color='pink', alpha=0.95)
        # Remove the black borders of the plot of the network
        [plt.axes().spines[side].set_color('white') for side in plt.axes().spines.keys()]

        # Define the layout of the title of the plot
        if 'P_k' not in parameters:  # For all the models except configuration model
            plt.title("Algorithm: " + alg_name + ". Parameters: " + str(parameters)[1:-1])
        else:  # For the configuration model (includes P_k)
            plt.title("Algorithm: " + alg_name + ". Parameters: 'N':" + str(parameters['N']) +
                      ", 'P_k': See at Distributions")

        plt.tight_layout()  # Avoid the lengthy titles to be cut

        # Show or save the plot
        if save_at is None:  # Show the plot if the title of the file is not given
            plt.show()
        else:  # Save in 'path' otherwise
            plt.savefig(os.path.join(save_at, 'Graph'))

        plt.close()

    def save_graph(self, path):
        """
        Saves the networkx graph stored in self.graph.

        :param path: Location to store the network.
        """

        # Save the network self.graph in 'path' with the name 'Graph'
        nx.write_pajek(G=self.graph, path=os.path.join(path, 'Graph'))
