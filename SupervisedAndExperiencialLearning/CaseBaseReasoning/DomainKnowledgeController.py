"""
Representation of the domain knowledge. This knowledge is represented through a Tree graph which was handcrafted
"""

import os
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

DB_FOLDER = 'CaseLibrary'
DOMAIN_KNOWLEDGE_PATH = os.path.join(DB_FOLDER, 'domain_knowledge.net')
CHILD_POS = 0

class DomainKnowledge:
    def __init__(self, domain_knowledge_path = DOMAIN_KNOWLEDGE_PATH):
        """
        Manager of the domain knowledge, offers all the functions relatives for transforming this knowledge in
        dissimilarities calculations and NLP related simmilarities.
        :param domain_knowledge_path: str. Path where the Pajek file contaning the tree network is saved
        """
        # Read the tree and verify that it is effectively a tree graph
        self.domain_knowledge_tree =nx.read_pajek(domain_knowledge_path)
        if not nx.is_tree(self.domain_knowledge_tree):
            raise AttributeError("Domain Knowledge must be a Tree")
        # Save a free graph version of this tree
        self.domain_knowledge_free_graph = nx.Graph(self.domain_knowledge_tree)
        # Stablish a list of known ingredients
        self.known_ingredients = np.unique(list(self.domain_knowledge_tree.nodes))
        # Consider as maximum dissimilarity allowed the diameter of the tree (it amount of levels)
        self.max_dissimilarity = nx.diameter(self.domain_knowledge_free_graph)


    def get_most_similar_ingredients(self, ingredients_requested, ingredients_available):
        """
        Get the most similar ingredient on the ingredients_available list for each ingredient on the ingredients_requested
        list.
        :param ingredients_requested: List of str. List of ingredients which are requested by the algorithm
        :param ingredients_available: List of str. List of ingredients where we want to search the similars
        :return: List of str. List of the most similar ingredient on the ingredients_available list for each
            ingredient on the ingredients_requested
        """

        description_requested = list(ingredients_requested)
        output = []
        while len(description_requested) > 0:
            # For each ingredient in the requested description get the minimum dissimilarity with another ingredient on the
            # found description (The dissimilarity to its most similar ingredient)
            dissimilarities = [[self.ingredient_dissimilarity(ingredient1=req, ingredient2=found)
                                for found in ingredients_available] for req in description_requested]
            correspondences = np.argmin(dissimilarities, axis=1)
            used_elements, offset = [], 0
            for i in range(len(correspondences)):
                if correspondences[i] not in used_elements:
                    output.append(ingredients_available[correspondences[i]])
                    used_elements.append(correspondences[i])
                    description_requested.pop(offset)
                else:
                    offset += 1

            ingredients_available = np.delete(ingredients_available, used_elements)
        return output

    def know_ingredient(self, ingredient):
        """
        Returns if an ingredient is known by the domain knowledge or not
        :param ingredient: str. Ingredient to test
        :return: Boolean. True if the ingredient is in the domain knowledge, False if it is not.
        """
        return ingredient in self.domain_knowledge_tree

    def ingredient_dissimilarity(self, ingredient1, ingredient2):
        """
        Get the dissimilarity between two ingredients based in the knowledge of the domain
        :param ingredient1: str. First ingredient for calculating the dissimilarity
        :param ingredient2: str. Second ingredient for calculating the dissimilarity
        :return: Int. Dissimilarity between both ingredients. It is interpreted as the shortest path which connects both
        nodes on the tree. (Dissimilarity between 'A' and 'A' is 0.).
        """
        # If both ingredients are within the domain return the len of its shortest path
        if self.know_ingredient(ingredient=ingredient1) and self.know_ingredient(ingredient=ingredient2):
            return len(nx.shortest_path(G=self.domain_knowledge_free_graph, source=ingredient1, target=ingredient2)) - 1
        # If any ingredient is not in known return the maximum dissimilarity
        else:
            return self.max_dissimilarity

    def fill_vector_with_all_parents(self, nodes):
        """
        In a recursive way, generate from each node in the vector nodes the full path of its parents
        :param nodes: List of str. List of ingredients from which generate all the predecessors path of each one
        :return: List of list of str. A vector which contains for each node in the vector nodes the full path of its parents
        """

        def all_predeccesors(node):
            """
            Recursive function which generates a vector with all the parents of a given nod
            :param node: str. Node of the tree
            :return: List of str. List with all the parents of the given node
            """
            # Take the predecessor of the current node and append it in a list
            predecessors = list(self.domain_knowledge_tree.predecessors(node)) if node in self.domain_knowledge_tree else []
            # If the node is not the root (which not have predecessors)
            if len(predecessors) > 0:
                # Extend the list with the rest of its predecessors
                predecessors.extend(all_predeccesors(predecessors[0]))
            # Return the list with all the predecessors of the current node
            return predecessors

        # For each node in the vector nodes, extend it with the full path of its parents
        return [[child] + all_predeccesors(node=child) for child in nodes]

    def plot_tree(self, layout=nx.planar_layout, save_at=None, file_name='Graph', print_labels=True):
        """
        Plots (or saves) the network stored in self.graph.

        :param layout: nx.Layout. Layout of the nodes of the represented network.
        :param save_at: String. Location to store the plot generated by this function.
        :param file_name: String. Name of the file where saving the graph plot
        :param print_labels: Boolean. If print the labels over the nodes or not
        """
        nx.draw_networkx(self.domain_knowledge_tree, pos=layout(self.domain_knowledge_tree), with_labels=print_labels, node_color='pink', alpha=0.8, font_size=8)
        # Remove the black borders of the plot of the network
        [plt.axes().spines[side].set_color('white') for side in plt.axes().spines.keys()]

        plt.title("Domain Knowledge")

        plt.tight_layout()  # Avoid the lengthy titles to be cut

        # Show or save the plot
        if save_at is None:  # Show the plot if the title of the file is not given
            plt.show()
        else:  # Save in 'path' otherwise
            if not os.path.isdir(save_at):
                os.makedirs(save_at)
            plt.savefig(os.path.join(save_at, file_name + '.png'), dpi=440)

        plt.close()

