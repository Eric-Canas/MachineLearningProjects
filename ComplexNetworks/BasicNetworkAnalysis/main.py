from Analyzer import Analyzer
from Printer import *
import os

if __name__ == '__main__':
    #Defines the directory information where files are located
    root = 'A1-networks'
    dir = 'real'
    full_dir = os.path.join(root, dir)
    """
    #Comment/Uncomment it for extracting network Descriptors
    #-----------------------------Network Descripors section-----------------------------
    #Calculate the network descriptors for the files within dir directory
    network_descriptors = [Analyzer(dir=dir, file=file).get_numeric_descriptors()
                           for file in os.listdir(full_dir)]
    #Print this descriptors in latex table format
    print("Latex Table of Network Descriptors for all networks in dir: " + dir)
    print_latex_table(dictionaries=network_descriptors,
                      net_names=[name[:len('.net')] for name in os.listdir(full_dir)])
    """

    #Defines the file where network is located for extracting node information and plots
    file = 'dolphins.net'
    # -----------------------------Node Descripors section-------------------------------
    #Comment/Uncomment it for extracting network Descriptors
    #Select the file where calculate the node descriptors
    #Calculate the descriptors of each node within the network dir/file
    node_data = Analyzer(dir=dir, file=file).get_node_descriptors()
    #Save all the nodes information into a CSV. Saved into results directory
    print_nodes_csv(dictionary=node_data,precision=8, title=file[:-len('.net')]+'_nodes.csv')
    #Print the selected nodes as a table in latex format (Modifiying it requires knowledge about nodes within the network.
    nodes_to_print = ['Beak','Bumper']
    print_latex_table(dictionaries=[node_data[node] for node in nodes_to_print], column_names=nodes_to_print,
                      decimals=8, measuring = 'Descriptor/Node')

    # --------------------------------Plotting section----------------------------------
    #Comment/Uncomment it for see the plots of the degree distributions
    #Plots the PDF and the Cumulative Complementary Degree Distribution for dir/file network
    #First as normal scale distributions and them as log-log distributions
    #Analyzer(dir=dir, file=file).plot_distribution(xlog=False, ylog=False, file=file[:-len('.net')].title())
    Analyzer(dir=dir, file=file).plot_distribution(xlog=True, ylog=True, file_name=file[:-len('.net')].title())