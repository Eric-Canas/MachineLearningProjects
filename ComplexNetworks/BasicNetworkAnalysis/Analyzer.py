import networkx as nx
import os
import numpy as np
from matplotlib import pyplot as plt

class Analyzer:
    """
    Read or takes a network and allows you to extract general analysis about it
    """
    def __init__(self, graph=None, root='A1-networks', dir = 'toy', file='circle9.net'):
        """
        If a graph in networkx format is taken initialize the object with it, if not, reads
        the graph from the file 'file' into directory 'root/dir/file'
        :param graph: Graph in a networkx format
        :param root: Root directory where graphs dirs are located
        :param dir: Graph directory within Root where network files are located
        :param file: Name of the network file to open
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
        Degree assortativity, Average path length & Diameter.
        :return:
        Dictionary with all the named network numeric descriptrs
        """

        descriptors = {}

        #Gets counts of nodes and edges of the network
        descriptors['Number of nodes'] = self.graph.number_of_nodes()
        descriptors['Number of edges'] = self.graph.number_of_edges()
        #Gets descriptors about the degree of his nodes (max, min and average)
        descriptors['Max degree'] = self.degree_descriptors(descriptor='max')
        descriptors['Min degree'] = self.degree_descriptors(descriptor='min')
        descriptors['Average degree'] = self.degree_descriptors(descriptor='average')
        #Gets the average clustering coefficient
        descriptors['Average clustering coefficient'] = nx.average_clustering(G=nx.Graph(self.graph))
        #Gets the degree assortativity
        descriptors['Degree assortativity'] = nx.degree_assortativity_coefficient(self.graph)
        #Gets the average shortest path length of the graph
        descriptors['Average path length'] = nx.average_shortest_path_length(self.graph)
        #Gets the diameter of the graph
        descriptors['Diameter'] = nx.diameter(self.graph)

        return descriptors

    def degree_descriptors(self, descriptor):
        """
        Gets the max, min or average degree description
        :param descriptor: 'max', 'min' of 'average'. Descriptor to extract
        :return:
        Integer value for min or max. Float value for average.
        """
        #Read the degree of nodes in the graph as a dtype named array (first col: node, second col: degree).
        degrees = np.array(list(self.graph.degree), dtype=[('node', '<U128'), ('degree', np.float)])
        #Gets a dictionary with the numpy function which corresponds to each posible descriptor value
        func = {'max': np.max, 'min': np.min, 'average': np.mean}
        #Applies the solicited function to the degree column of the array
        return func[descriptor](degrees['degree'])

    def get_node_descriptors(self):
        """
        Gets the node descriptors for each node in the graph
        :return:
        A dictionary of dictonaries where first key is node name and second key the descriptor name
        """
        #First execute the algorithms where the full graph must be analyzed
        #(Not sense of analyzing a single node)
        betweenness = nx.betweenness_centrality(self.graph)
        eigenvector_crentrality = nx.eigenvector_centrality(nx.Graph(self.graph))
        pagerank = nx.pagerank(nx.Graph(self.graph))

        #Generates the dictionary of the first key (Node name)
        node_descriptors = {}
        for node in self.graph.nodes:
            #Generates the dictionary of the second key (Descriptor)
            node_descriptors[node] = {}
            #Gets the degree of the node
            node_descriptors[node]['Degree'] = self.graph.degree[node]
            #Gets the strength of the node (The sum of his weights)
            node_descriptors[node]['Strength'] = np.sum([weight for _, _, weight in self.graph.edges(node, 'weight')])
            #Gets the clustering coefficient
            node_descriptors[node]['Clustering coefficient'] = nx.clustering(nx.Graph(self.graph), node)
            #Gets the average path length (By the mean of the shortes path length between the node and the rest of the graph)
            node_descriptors[node]['Average path length'] = np.mean(list(nx.shortest_path_length(self.graph, node).values()))
            #Gets his maximum path length (Same as the eccentricity of the node)
            node_descriptors[node]['Maximum path lenght'] = nx.eccentricity(self.graph, v=node)
            #For the previously calculated algorithm takes the value of the current node.
            node_descriptors[node]['Betweenness'] = betweenness[node]
            node_descriptors[node]['Eigenvector centrality'] = eigenvector_crentrality[node]
            node_descriptors[node]['PageRank'] = pagerank[node]

        #Return the dictionary of dictionaries
        return node_descriptors

    def plot_distribution(self, bins=None, log_bins=10, xlog=False, ylog=False, file_name=None):
        """
        Plots the Probability Degree Distribution and the Complementary Cumulative Degree Distribution
        :param bins: Number of bins to use in the case of not using a log scale
        :param log_bins: Number of logarithmic bins to use in the case where using a log scale
        :param xlog: Apply logscale into x axis?
        :param ylog: Apply logscale into y axis?
        :param file_name: Name of the file (If given add it to the title of the plot)
        :return:
        None
        """
        #Get the degrees of all the nodes in the graph (Used for making the histogram of degrees)
        degrees = np.array([d for _, d in self.graph.degree()])

        #Decide how to divide the bins
        if xlog:
            #Divide in a logarithmic scale between 0 and the max degree
            bins = np.logspace(np.log10(degrees.min())-0.0001, np.log10(degrees.max()), log_bins)
            #Correcting the log(0) = 1 definition

        if bins is None:
            #Divide into a equally divided scale between 0 and the max with as many bins as max degree
            bins = np.arange(degrees.max()+1)+0.0001

        #Calculates the histogram with the given bins
        histogram, bins = np.histogram(degrees, bins=bins)
        #Transform the histogram into a probability distribution (making it to sum 1)
        histogram = histogram/np.sum(histogram)

        #Plots the PDF and CCDD
        for title in ('Probability Degree Distribution', 'Complementary Cumulative Degree Distribution'):
            #CCDD is the complementary of CDD (1-CDD). CDD is the cumulative sum of the PDF
            if 'Cumulative' in title:
                histogram = 1-np.cumsum(histogram)
            #Select the scale of the plots
            if xlog:
                plt.bar(bins[1:], histogram, width=[bins[i]-bins[i-1] for i in range(1, len(bins))])
                plt.xscale('log')
            if ylog:
                plt.yscale('log')
            #If Y axis scale is not logarithmic plot a bars graphic, else a line graphic
            else:
                plt.bar(bins[1:], histogram)
            #Add the name of the network to plot title if given
            if file_name is not None:
                title += ' (' + file_name + ' Network)'

            #Plot titles and name of the axis
            plt.title(title)
            plt.xlabel('Degree')
            plt.ylabel('Probability')
            #Show the plot
            plt.show()
