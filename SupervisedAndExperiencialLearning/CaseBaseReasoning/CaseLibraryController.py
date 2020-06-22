"""
This file represents and controls the Case Library. The CaseLibrary is the class which should be used by the application,
is the controller of the database. The database clase represents the low lever database which is controlled by the CaseLibrary.
"""

import os
import numpy as np
import nltk
import xml.etree.ElementTree as ET
from xml.dom import minidom
import warnings

# ------------------------------------ PATHS AND IDENTIFIERS DEFINITIONS --------------------------------------------
DB_FOLDER = 'CaseLibrary'
FIRST_CASE_LIBRARY_PATH = os.path.join(DB_FOLDER, 'ccc_sandwich.xml')
UPDATED_CASE_LIBRARY_PATH = os.path.join(DB_FOLDER, 'updated_ccc_sandwich.xml')
DOMAIN_KNOWLEDGE_PATH = os.path.join(DB_FOLDER, 'domain_knowledge.net')
IDENTIFIER, DERIVED_FROM, DESCRIPTION, SOLUTION, EXCEPTIONAL, USABILITY = 'Identifier','Derived From', 'Description', 'Solution', 'Is Exceptional', 'Usability'
MAX, MIN = 1,0

# --------------------------------------------- PARAMETERS --------------------------------------------------------
# Dissimilarity parameters
LENGHT_PENALIZER = 0.7
INGREDIENTS_PENALIZER = 0.3
# Relevance parameters
KNN3_WEIGHT, KNN1_WEIGHT, USABILITY_WEIGHT = 0.3, 0.6, 0.1
# Learning parameters
SIGMA = 1.5
MINIMUM_RELEVANCE_FOR_LEARNING = 0.5
MINIMUM_RELEVANCE_FOR_FORGETTING = 0.5
# Forgetting Parameters
USABILITY_DECREASING = 0.01
MAX_USABILITY = 1.



class CaseLibrary:

    def __init__(self, domain_knowledge, case_library_path = FIRST_CASE_LIBRARY_PATH, new_db_path = UPDATED_CASE_LIBRARY_PATH):
        """
        Case Library Controller which will manage the database
        :param domain_knowledge: DomainKnowledge Object. Manages the all the referent to the DomainKnowledge
        :param case_library_path: str. Path where the physical case libray xml is.
        :param new_db_path: str. Path where the Case Library xml will be save when flushing
        """
        self.domain_knowledge = domain_knowledge
        self.database = Database(db_tree=ET.parse(case_library_path).getroot(), domain_knowledge=self.domain_knowledge)
        self.new_db_path = new_db_path

    def check_ingredients_validity(self, ingredients):
        """
        Given a set of ingredients check if they are known by the domain knowledge
        :param ingredients: List of str. Ingredients for checking
        :return: boolean. If it is a valid or invalid list of ingredients
        """
        return np.all(np.isin([ingredient.lower() for ingredient in ingredients], self.domain_knowledge.known_ingredients))

    def propose_ingredient_mistake_solutions(self, ingredients):
        """
        Takes an input array of ingredients, and generate a new one with a new proposal for those ingredients
        which did not match due a typo. Example: ["Ham", "Tomtoes"] ["Ham", "Tomato"]. They are computed through
        edit distances in order to search for spells mistakes
        :param ingredients: List of str. Ingredients containing mistakes
        :return: List of str. The new proposition for these ingredients.
        """
        ingredients_proposition = []
        # For each one of the ingredients
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            # If it already exists uses the same
            if ingredient_lower in self.domain_knowledge.known_ingredients:
                ingredients_proposition.append(ingredient)
            # If not, substitute it for the most similar word (the one with lowest edit distance)
            else:
                distances = [nltk.edit_distance(s1=ingredient_lower, s2=ingredient_db) for ingredient_db in self.domain_knowledge.known_ingredients]
                ingredients_proposition.append(self.domain_knowledge.known_ingredients[np.argmin(distances)].title())

        return ingredients_proposition

    def select_best_case(self, promising_cases):
        """
        Select the best case among cases. In this case, as we can ensure that all cases are promising cases, thus all of them
        fits as much as possible the user requirements, and all features are categorical (ingrendients), the best case will
        be the one with less additional ingredients, since it will be the more similar to the original proposal. If there is
        more than one promissing case with the same amount of ingredients, then select the simplest one (the one with less steps).
        (Ockham's Razor)
        :return:
            The instance of promising_cases containing the best case.
        """
        # Put it as numpy array for efficient and easy multi-indexation
        promising_cases = np.array(promising_cases, dtype=np.object)
        # Take the ingredients length of all promising cases
        ingredients_length = np.array([len(case[DESCRIPTION]) for case in promising_cases])
        # Select the cases (among the promising ones) which have the lower amount of ingredients
        best_cases = promising_cases[ingredients_length == np.min(ingredients_length)]
        # If there is only one case with that minimal amount of ingredients return it
        if len(best_cases) == 1:
            return best_cases[0]
        # If there are more than one, return among them the one which have a lower amount of steps (heuristically the simplest one)
        else:
            steps = [len(nice_case[SOLUTION]) for nice_case in best_cases]
            return best_cases[np.argmin(steps)]


    def retrieve(self, desired_description, return_promising_dissimilarities=True):
        """
        Get the best case within the database, for this purpose, it request to the database for the promising cases.
        These are the cases which have the lowest description dissimilarity with the new requiered description. Among
        them apply heuristics for giving a best case (The one with lowest amount of additional ingredients), and if there
        is a tie, the one with the simplest solution. Moreover, as this function is the one which search through the network,
        each time is called decrease the usability of all non retrieved cases and maximizes the usability of the used one.
        :param desired_description: List of str. List of ingredients of the new recipe
        :return: Dictionary. Best case
        """

        # First, retrieve from the database the set of promissing cases
        promissing_cases = self.database.retrieve_most_similar_cases(prefered_description=desired_description)

        # Then, select the best case among them
        best_case = self.select_best_case(promising_cases=promissing_cases)

        # Decrease the usability of all cases
        self.decrease_usabilities(decrement=USABILITY_DECREASING)
        # Revive the usability of the best case
        self.database.revive_usability(case_id=best_case[IDENTIFIER], revive_to=MAX_USABILITY)

        # If the algorithm was asked for also retrieving the dissimilarity of the promissing cases (usually for
        # verbosing purposes, calculate it and return too)
        if return_promising_dissimilarities:
            return best_case, [np.round(self.database.dissimilarity(description_requested=desired_description,
                                                                    description_found=promissing_case[DESCRIPTION]),
                                        decimals=2) for promissing_case in promissing_cases]
        # Else, return only the best case
        else:
            return best_case

    def compose_case_from_parameters(self, identifier, derived_from, description, solution, exceptional=False):
        """
        Gives a dictionary representing the case defined by the function parameters. This function only calls to its
        counterpart function in low level.
        :param identifier: str. Identifier of the recipe
        :param derived_from: str. Identifier of the recipe from which it has been derived (or itself if is not derived)
        :param description: list of str. Description of the problem (The set of ingredients of the recipe)
        :param solution: list of str. The solution of the problem (The list of steps of the recipe)
        :param exceptional: boolean. If it is an exceptional case (True) or an usual case (False)
        :return:
            Dictionary. The representation of the case as a dictionary
        """
        return self.database.compose_case_from_parameters(identifier=identifier, derived_from=derived_from,
                                                   description=description, solution=solution, exceptional=exceptional)

    def compose_new_identifier(self, base_identifier, new_ingredients):
        """
        Generate a new identifier from a base identifier of a case, having in account that it uses now a different
        set of ingredients.
        :param base_identifier: str. Identifier of the case from which the new recipe has been adapted
        :param new_ingredients: List of str. List of ingredients that the new recipe uses
        :return:
            str. The new identifier of the case
        """
        return self.database.compose_new_identifier(base_identifier=base_identifier, new_ingredients=new_ingredients)

    def save_case(self, case, verbose=True):
        """
        Save a case in the database
        :param case: Dictionary. Definition of a case.
        :return
            Boolean. True if the case was correctly saved, False if not.
        """
        # Save the case
        saved = self.database.save_case(case=case)

        # Verbose that it was saved or that it was not saved if a failure ocurred
        if verbose:
                print("Case "+('' if saved else 'not')+" saved")

        return saved

    def flush_database(self):
        """
        Save the database in an XML file (Defined by new_db_path attribute of the Case Library)
        """
        self.database.flush(xml_file=self.new_db_path)

    def propose_for_learning(self,case, verbose = True):
        """
        Propose a case for be learned if it is considered specially relevant (or if it is exceptional)
        :param case: Dictionary. Definition of the new case which is proposed for being learned
        :param verbose: Boolean. Verbosing flag
        :return: boolean. True if case has been learned, False otherwise
        """
        # If the case is not exceptional calculate its relevance (also if there is verbosing required)
        if not case[EXCEPTIONAL] or verbose:
            relevance_of_case = self.database.relevance_of_case(case=case, is_in_train=False)
            if verbose:
                print(("Expected r" if case[EXCEPTIONAL] else "R")+"elevance of this new case in the database: "+
                      str(np.round(relevance_of_case, decimals=2)))
                if case[EXCEPTIONAL]:
                    print("However, it is an exceptional case")

        # If case was exceptional or if it was enough relevant: Learn it.
        if case[EXCEPTIONAL] or relevance_of_case > MINIMUM_RELEVANCE_FOR_LEARNING:
            # Save at database
            saved = self.database.save_case(case=case)
            if verbose:
                if saved:
                    print("Learning this new case")
                else:
                    warnings.warn("Case was not learned due to an error")
            return saved
        # Otherwise skip it
        elif verbose:
            print("This case is not enough relevant for be learned")
            return False

    def forget_step(self, verbose = True):
        """
        Develop the complete forgetting step. Searching on the database for unrelevant cases and forgetting
        the enough unrelevant ones (considering its usability). This forgetting step will recompute all the relevances
        each time a case is forgotten.
        :param verbose: Verbosing flag
        :return: boolean. If anything was forgotten
        """
        # For simulating a do-while loop set the first flag
        forget_step_ended,anything_forgotten  = False, False
        # While there are forgettable cases
        while not forget_step_ended:
            # For each case in the database
            for i in range(len(self.database)):
                # Take the case and compute its relevance
                case = self.database.compose_case_dictionaries(cases=[i])[0]
                relevance = self.database.relevance_of_case(case=case,is_in_train=True)

                # If its relevance is under a threshold forget it
                if relevance < MINIMUM_RELEVANCE_FOR_FORGETTING:
                    self.database.forget_case(case_id=i)
                    # If verbosing print the forgetting reasons
                    if verbose:
                        self.verbose_forgetting_reasons(case=case, relevance=relevance, case_id=i)
                        self.verbose_KNN_dissimilarities(new=False)

                    # Recompute the stathistics of the network for calculating relatives relevances
                    self.database.recompute_similarity_parameters()
                    if verbose:
                        self.verbose_KNN_dissimilarities(new=True)
                    # If anything has been forgotten record it and restart the bucle
                    anything_forgotten = True
                    break
            else:
                forget_step_ended = True

        return anything_forgotten

    def decrease_usabilities(self, decrement=USABILITY_DECREASING):
        """
        Decrease the usability of all the instances in the dataset
        :param decrement: float: Decreasing amount
        """
        self.database.decrease_usabilities(decrement=decrement)

    def verbose_KNN_dissimilarities(self, new):
        """
        Verbose the current stathistics of the Case Library
        :param new: Boolean. If they are the new or the old stathistics (Before or after an update)
        """
        print(("Now the" if new else "The")+" 3 Nearest Neighbor mean dissimilarity "+("is" if new else "was")+": "
              + str(np.round(self.database.dissimilarity3_mean, decimals=2)) +
              ", with an standard deviation of: " +
              str(np.round(self.database.dissimilarity3_std, decimals=2)))
        print(("Now the" if new else "The")+" 1 Nearest Neighbor mean dissimilarity "+("is" if new else "was")+": "
              + str(np.round(self.database.dissimilarity1_mean, decimals=2)) +
              ", with an standard deviation of " +
              str(np.round(self.database.dissimilarity1_std, decimals=2)))

    def verbose_forgetting_reasons(self, case, relevance, case_id):
        """
        Print the reasons why a case was forgetted
        :param case: Dictionary. Dictionary which describes the case
        :param relevance: float. Relevance of the case
        :param case_id: int. Id of the case in the dataset
        :return:
        """
        print("\nThe following case has been forgotten because of its low relevance: " + str(
            np.round(relevance, decimals=2)))
        print(recipe_to_txt(recipe=case))
        similar_recipes = self.database.retrieve_most_similar_cases(prefered_description=case[DESCRIPTION], case_id=case_id)
        print("It is quite similar into its description to the following case" + (
            's:' if len(similar_recipes) > 1 else ':'))
        for similar_recipe in similar_recipes:
            print("\n" + recipe_to_txt(recipe=similar_recipe))


# Database Represents the low level part of the Case Library, only the CaseLibrary controller class should access to it
class Database:
    def __init__(self, db_tree, domain_knowledge):
        """
        Composes an internal database in the most efficient way for flat searches. This method is separated arrays
        which shares indexes. In this way all of them can be efficiently iterated, and referred by its index identifier.
        Moreover, searches and massive comparisons can be efficiently done through numpy methods which are implemented in C.
        :param db_tree. ElementTree Tree XML tree defining the physical database
        :param domain_knowledge. DomainKnowledge Object. Object defining the domain knowledge.
        """

        self.domain_knowledge = domain_knowledge
        # Define the attributes of cases
        self.identifiers, self.derivation, self.description, self.exceptionality, self.solution, self.usability = [], [], [], [], [], []

        #Initialize the attributes of each case
        for recipe in db_tree.findall('./recipe'):
            # Set the name of the recipe as identifier
            self.identifiers.append(recipe.find('title').text.lower())
            # As it was an original case, it is derived from itself
            self.derivation.append(recipe.find('title').text.lower())
            # Save the descriptor of their ingredients as the recipe definition
            self.description.append([ingredient.attrib['food'].lower() for ingredient in recipe.findall('./ingredients/ingredient')])
            # Save the equivalence of tokens with the ones which are used in the natural language steps of the solution
            self.exceptionality.append((bool(int(recipe.find('exceptional').text.lower())) if recipe.find('exceptional') is not None else False))
            # Save all the steps of the solution
            self.solution.append([step.text.lower() for step in recipe.findall('./preparation/step')])
            # Save the usability of the case
            self.usability.append((float(recipe.find('usability').text.lower()) if recipe.find('usability') is not None else 1.))

        # Put as numpy array those arrays which will need massive comparisons or indexing
        self.identifiers = np.array(self.identifiers, dtype='<U128')
        self.derivation = np.array(self.derivation, dtype='<U128')

        # Comput the stathistics of the network
        self.recompute_similarity_parameters()

    def __len__(self):
        """
        Overrides the Len function defining the Length of the database.
        :return: Length of the database
        """
        return len(self.identifiers)

    def get_case(self, numeric_id):
        """
        Get the dictionaries of a case or cases by its numeric index
        :param numeric_id: int or list of ints. Index or indexes of the cases for return
        :return: Dictionary or list of Dictionaries. The recipes well represented as dicts.
        """
        if type(numeric_id) is int:
            return self.compose_case_dictionaries([numeric_id])[0]
        elif type(numeric_id) in (list, tuple, np.ndarray):
            return self.compose_case_dictionaries(numeric_id)
        else:
            raise ValueError("numeric_id can not be of "+str(type(numeric_id))+". Only integer or list types allowed")
    def recompute_similarity_parameters(self, sigma = SIGMA):
        """
        Compute and initilaze (or update) the similarity parameters of the Case Library. These parameters are computed
        without using the exceptional cases. These parameters are: The mean and std of the vectors containing the
        3 nearest neighbor and nearest neighbor dissimilarities for all the cases and the relevance ranges.
        These ranges are included between the mean-sigma*std (clipped at 0) and mean+sigma*std, being sigma a parameter.
        These ranges will relativice dissimilarities to the mean dissimilarity of the database, for detecting cases specially
        relevants or unrelevant.
        :param float. Sigma parameter explaining the acceptation of the deviation from the std.
        """
        # Compute the vector of dissimilarities for all not exceptional cases
        KNN3_dissimilarities, min_dissimilarity = np.array([self.KNN_dissimilarity(description=description, k=3,
                                                                                   is_in_train=True, return_min=True)
                    for description, exceptional in zip(self.description, self.exceptionality) if not exceptional]).T

        # Compute their means and standard deviations
        self.dissimilarity3_mean, self.dissimilarity3_std = np.mean(KNN3_dissimilarities), np.std(KNN3_dissimilarities)
        self.dissimilarity1_mean, self.dissimilarity1_std = np.mean(min_dissimilarity), np.std(min_dissimilarity)

        # Compute the ranges for considering the relative relevance
        self.dissimilarity3_relevance_range = (min(self.dissimilarity3_mean - sigma * self.dissimilarity3_std, 0),
                                               self.dissimilarity3_mean + sigma * self.dissimilarity3_std)
        self.dissimilarity1_relevance_range = (min(self.dissimilarity1_mean - sigma * self.dissimilarity1_std, 0),
                                               self.dissimilarity1_mean + sigma * self.dissimilarity1_std)


    def compose_case_dictionaries(self, cases):
        """
        Return the cases determined by indexes in cases as a list of dictionaries.
        :param cases: list of int. List of indexes of the cases to compose.
        :return:
            List of dictionaries. The representation of the cases as a dictionaries.
        """
        return [{IDENTIFIER: self.identifiers[case],
                  DERIVED_FROM: self.derivation[case],
                  DESCRIPTION: self.description[case],
                  SOLUTION: self.solution[case],
                  EXCEPTIONAL: self.exceptionality[case],
                  USABILITY: self.usability[case]}
                for case in cases]
    def compose_case_from_parameters(self, identifier, derived_from, description, solution, exceptional = False, usability=MAX_USABILITY):
        """
        Gives a dictionary representing the case defined by the function parameters
        :param identifier: str. Identifier of the recipe
        :param derived_from: str. Identifier of the recipe from which it has been derived (or itself if is not derived)
        :param description: list of str. Description of the problem (The set of ingredients of the recipe)
        :param solution: list of str. The solution of the problem (The list of steps of the recipe)
        :param exceptional: boolean. If it is an exceptional case (True) or an usual case (False)
        :return:
            Dictionary. The representation of the case as a dictionary
        """
        return {IDENTIFIER: identifier,
                  DERIVED_FROM: derived_from,
                  DESCRIPTION: description,
                  SOLUTION: solution,
                  EXCEPTIONAL: exceptional,
                  USABILITY: usability}

    def compose_new_identifier(self, base_identifier, new_ingredients):
        """
       Generate a new identifier from a base identifier of a case, having in account that it uses now a different
       set of ingredients.
       :param base_identifier: str. Identifier of the case from which the new recipe has been adapted
       :param new_ingredients: List of str. List of ingredients that the new recipe uses
       :return:
           str. The new identifier of the case
       """
        # Find the ingredients which the base recipe uses
        base_ingredients = np.array(self.get_case_by_identifier(identifier=base_identifier)[DESCRIPTION])
        # Find which are the news ingredients that the new recipe includes
        really_new_ingredients = new_ingredients[np.isin(new_ingredients, base_ingredients) == False]
        # If there are new ingredients, use them for generating the new identifier
        if len(really_new_ingredients) > 0 :
            new_indentifier = base_identifier+ ' with '+', '.join(really_new_ingredients)
        # If not, search if there is a lack of ingredients which difference them
        else:
            # Find the ingredients with the new recipe do not have
            lack_of_ingredients = base_ingredients[np.isin(base_ingredients, new_ingredients)]
            # If there is a lack of ingredients, use them for defining the new identifier
            if len(lack_of_ingredients)>0:
                new_indentifier = base_identifier+ ' without '+', '.join(lack_of_ingredients)
            # If both uses the same ingredients append an increasing identifier (version 2, 3...)
            else:
                new_indentifier = base_identifier+('I' if base_identifier[-1] ==('I') else '. II')
        return new_indentifier

    def get_case_by_identifier(self, identifier):
        """
        Get the dictionary of a case by its string identifier
        :param identifier: str. String identifier of the searched case
        :return: Dictionary. Dictionary representing the case which has 'identifier' as its identifier
        """
        return self.compose_case_dictionaries(np.where(self.identifiers == identifier)[0])[0]

    def retrieve_most_similar_cases(self, prefered_description, case_id = -1):
        """
        Retrieve the most similar cases for a given description
        :param prefered_description: List of str. Description of the new case. List of ingredients
        :param case_id: int. Index of the case if it was in training, used for not retrieving itself. Negative value
        if it was not in train
        :return: List of dictionaries. List of the most similar cases (promising cases) found for this case.
        """
        # Get the disimilarity from the given preferred description to the rest of the dataset
        distances = [self.dissimilarity(description_requested=prefered_description, description_found=ingredients)
                        for i, ingredients in enumerate(self.description)]

        # If the case was in training set its distance to infinite, for not taking it into account
        if case_id >= 0:
            distances[case_id] = np.inf
        # Get the indexes of all cases which minimizes the dissimilarity
        most_similar_cases = np.where(distances == np.min(distances))[0]
        # Return those cases as dictionaries
        return self.compose_case_dictionaries(cases=most_similar_cases)

    def dissimilarity(self, description_requested, description_found, length_difference_penalizer = LENGHT_PENALIZER,
                      ingredients_penalizer = INGREDIENTS_PENALIZER):
        """
        Defines the similarity between the ingredients requested and found. This disssimilarity between 2 ingredients
        is the distance between both nodes in the tree of the domain knowledge plus a penalizer that has in account the
        differences between the length of its descriptions.
        :param description_requested: List of str. Ingredients requested by the user (desired description).
        :param description_found: List of str. Ingredients found into the dataset (found description)
        :param length_difference_penalizer: How penalizes each ingredient more or less in the description
        :return: Float. Dissimilarity between both descriptions
        """

        #Get the lenght dissimilarity, if there is a lack of ingredients in the recipe found, return infinite
        if len(description_found) >= len(description_requested):
            length_dissimmilarity = len(description_found) - len(description_requested)
        else:
            return np.inf

        dissimilarity = 0
        description_requested = list(description_requested)
        while len(description_requested) > 0:
            # For each ingredient in the requested description get the minimum dissimilarity with another ingredient on the
            # found description (The dissimilarity to its most similar ingredient)
            dissimilarities = [[self.domain_knowledge.ingredient_dissimilarity(ingredient1=req, ingredient2=found)
                                     for found in description_found if self.domain_knowledge.know_ingredient(ingredient=found)]
                                    + [self.domain_knowledge.max_dissimilarity]
                             for req in description_requested]
            correspondences = np.argmin(dissimilarities, axis=1)
            used_elements, offset = [], 0
            for i in range(len(correspondences)):
                if correspondences[i] not in used_elements:
                    dissimilarity += dissimilarities[i][correspondences[i]]
                    used_elements.append(correspondences[i])
                    description_requested.pop(offset)
                else:
                    offset+=1

            description_found = np.delete(description_found, used_elements)

        # Get the sum of these minimum disimilarities plus a pondered penalization of the difference in lengths descriptions
        return ingredients_penalizer * dissimilarity + length_difference_penalizer * length_dissimmilarity

    def save_case(self, case):
        """
        Save a case
        :param case: Dictionary. Definition of a case.
        :return boolean. True if the case was trully save, False if it was not saved due to identifier duplicity
        """
        # For security, confirm that the unique restriction of the identifier is satisfied, if it is not, skip the saving
        if np.isin(case[IDENTIFIER],self.identifiers):
            warnings.warn("Attempting to save an already existent identifier, skipping the save")
            # Return Fail
            return False
        else:
            #Save all the information
            self.identifiers = np.concatenate([self.identifiers,[case[IDENTIFIER]]])
            self.derivation = np.concatenate([self.identifiers,[case[DERIVED_FROM]]])
            self.description.append(list(case[DESCRIPTION]))
            self.solution.append(case[SOLUTION])
            self.exceptionality.append(case[EXCEPTIONAL])
            self.usability.append(case[USABILITY])
            # Return Success
            return True

    def forget_case(self, case_id):
        """
        Forget an existent case
        :param case_id: int index of the case to forget
        """
        # Delete all the data of these attributes
        self.identifiers = np.delete(self.identifiers,case_id)
        self.derivation = np.delete(self.derivation, case_id)

        self.description.pop(case_id)
        self.solution.pop(case_id)
        self.exceptionality.pop(case_id)
        self.usability.pop(case_id)

    def KNN_dissimilarity(self, description, k=3, is_in_train=True, return_min=True, is_exceptional=False):
        """
        Get the K nearest neighbor dissimilarity between the given description and the rest of instances in the dataset
        :param description: List of st. Description of the case
        :param k: int. K of the K Nearest Neighbor algorithm
        :param is_in_train: Boolean. If the instance searched is already in the training dataset
        :param return_min: Boolean. If also return the 1 Nearest Neighbor result
        :param is_exceptional: If the case was exceptional or not
        :return: Float or tuple of floats. Mean K Nearest Neighbor Dissimilarity. If return_min, also return the
        Nearest Neighbor dissimilarity
        """
        # Set 1 if the case will be found in the database, but is not exceptional
        # (because dissimilarity of exceptional_cases is considered infinite)
        is_in_train = int(is_in_train and not is_exceptional)
        # Calculate the dissimilaritiy with all the database.
        dissimilarities = np.array([(self.dissimilarity(description_requested=description, description_found=instance)
                                     if not exceptionality else self.domain_knowledge.max_dissimilarity*len(description))
                for instance, exceptionality in zip(self.description, self.exceptionality)])

        # Get the K nearest neighbors dissimilarity (not including itself if was in the database
        k_most_similar = np.sort(dissimilarities[np.argpartition(dissimilarities, kth=is_in_train+k)[:is_in_train+k]])[is_in_train:]
        # Returns the K Nearest Neighbor mean dissimilarity, and if return_min, also the Nearest Neighbor dissimilarity
        if return_min:
            return np.mean(k_most_similar), np.min(k_most_similar)
        else:
            return np.mean(k_most_similar)


    def relevance_of_case(self, case, is_in_train=True, knn3_weight=KNN3_WEIGHT, knn1_weight=KNN1_WEIGHT,
                          usability_weight=USABILITY_WEIGHT):
        """
        Gets the relevance of a case. It is calculated using as reference the mean and std dissimilarities of the database.
        It ponders the 3 Nearest Neighbor mean dissimilarity, the dissimilarity with its Nearest Neighbor and the usability
        of the case.
        :param case: Dictionary. Description of the case.
        :param is_in_train: Boolean. If the case was alrealy in train
        :return: Relative measure of the relevance of the case within the network
        """
        # Calculates the dissimilarity with its 3 nearest neighbor and with it nearest neighbor
        dissimilarity_3, dissimilarity_1 = self.KNN_dissimilarity(description=case[DESCRIPTION], k=3, is_in_train=is_in_train,
                                                            return_min=True, is_exceptional=case[EXCEPTIONAL])
        # Calculate its relative relevance measure for 3 nearest neighbors and the nearesdt neighbor
        relevance_3 = max(dissimilarity_3/(self.dissimilarity3_relevance_range[MAX]-self.dissimilarity3_relevance_range[MIN]), 0)
        relevance_1=  max(dissimilarity_1 / (self.dissimilarity1_relevance_range[MAX] - self.dissimilarity1_relevance_range[MIN]), 0)

        #Ponders the 3 Nearest Neighbor mean dissimilarity, the dissimilarity with its Nearest Neighbor and the usability of the case.
        return knn3_weight * relevance_3 + knn1_weight * relevance_1 + usability_weight * case[USABILITY]

    def decrease_usabilities(self, decrement=USABILITY_DECREASING):
        """
        Decrease the usability parameter of each instance
        :param decrement: Amount of decrement
        """
        for i in range(len(self.usability)):
            self.usability[i] = max(self.usability[i]-decrement, 0)

    def revive_usability(self, case_id, revive_to = MAX_USABILITY):
        """
        Reset the usability of a concrete case. It should be called only when a case is used
        :param case_id: str or int. Identifier of a case of id
        """
        if type(case_id) in [str, np.str_]:
            self.usability[np.where(self.identifiers == case_id)[0][0]] = MAX_USABILITY
        elif type(case_id) is int:
            self.usability[case_id] = revive_to
        else:
            raise ValueError("Unrecognized type of case_id: "+str(type(case_id)))


    def flush(self, xml_file = UPDATED_CASE_LIBRARY_PATH):
        """
        Save the database
        :param xml_file: str. Path of the xml where save the new database
        """
        # Write the top element of the tree
        top = ET.Element("recipes")
        # For each recipe
        for id, derivation, ingredients, steps, exceptional, usability in \
                zip(self.identifiers, self.derivation, self.description, self.solution, self.exceptionality, self.usability):
            # Build the parent node of the recipe
            recipe = ET.SubElement(top, "recipe")

            # Build the title of the recipe
            title = ET.SubElement(recipe, "title")
            title.text = id.title()

            # Save from which recipe it was derived from
            derived = ET.SubElement(recipe, "derivation")
            derived.text = derivation.title()

            # Build the parent node of the ingredients
            ingredients_node = ET.SubElement(recipe, "ingredients")
            # For each ingredient
            for ingredient in ingredients:
                # Save as child of ingredients the concrete ingredient. As attribute and text of the tag
                ingredient_node = ET.SubElement(ingredients_node, "ingredient",attrib={'food':ingredient.title()})
                ingredient_node.text = ingredient.title()

            # Build the parent node of the solution
            steps_node = ET.SubElement(recipe, "preparation")
            # For each step in the solution
            for step in steps:
                # Save it as child of the solution tag (preparation)
                step_node = ET.SubElement(steps_node, "step")
                step_node.text = step.capitalize()

            # Save if it was an exceptional case or not. As attribute and text
            exceptional = ET.SubElement(recipe, "exceptional", attrib={'is':str(int(exceptional))})
            exceptional.text = str(int(exceptional))

            # Save the current usability of the case
            usable = ET.SubElement(recipe, "usability")
            usable.text = str(usability)


        # Convert the tree to a string pretty printed
        pretty_tree = minidom.parseString(ET.tostring(top)).toprettyxml(indent="    ")
        # Save it at the xml file
        with open(xml_file, "w") as f:
            f.write(pretty_tree)







def recipe_to_txt(recipe):
    """
    Convert a recipe represented in a Dictionary to an string for pretty printing it
    :param recipe: Dictionary. Representation of the recipe.
    :return: str. String representing the dictionary pretty printed
    """
    # ...----- Recipe ------...
    txt = '-'*30+ "RECIPE"+'-'*30+"\n"
    # For each item
    for key, value in recipe.items():
        # Print the name
        txt += str(key)+': '
        # If its value is a list
        if type(value) in [list, np.ndarray]:
            txt += '\n'
            # Print it as enumeration
            for i, element in enumerate(value):
                txt+= str(i)+': '+str(element.capitalize())+'\n'
        # If it was an unique non iterable value
        else:
            #Print it directly
            txt+=str(value).title()
        txt+='\n'
    # ...-----------...
    txt += '-'*66
    return txt

