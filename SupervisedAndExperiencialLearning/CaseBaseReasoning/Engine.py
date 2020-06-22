"""
Engine of the Case Base Reasoning system, it will launch and manage the system.
"""
import numpy as np
from CaseLibraryController import CaseLibrary, IDENTIFIER, DERIVED_FROM, FIRST_CASE_LIBRARY_PATH,\
    UPDATED_CASE_LIBRARY_PATH, SOLUTION, DESCRIPTION,EXCEPTIONAL, recipe_to_txt
from DomainKnowledgeController import DomainKnowledge,  DOMAIN_KNOWLEDGE_PATH
from SolutionAdapter import adapt_solution, find_ingredients_used_by_step, verbose_step_matches

MIN_RATE, MAX_RATE = 0.,5.
MAX_INGREDIENTS, MAX_STEPS = 10, 50 # The four most large descriptions within the database uses 10 ingredients.
INGREDIENT_POS, REFERENCED_AS_POS = 0, 1

class Engine:

    def __init__(self, verbose=True, domain_knowledge_db_path=DOMAIN_KNOWLEDGE_PATH, case_library_path = FIRST_CASE_LIBRARY_PATH):
        """
         Class which will manage the complete engine of the case base reasoning system
        :param verbose: Boolean. If True, all actions of the  system will be verbosed. Default: True.
        :param domain_knowledge_db_path: Path of the Domain Knowledge network. Default: Predifined path.
        :param case_library_path: Path of the case library database. Default: Predifined path.
        """
        self.domain_knowledge = DomainKnowledge(domain_knowledge_path=domain_knowledge_db_path)
        self.case_library = CaseLibrary(domain_knowledge=self.domain_knowledge,case_library_path=case_library_path)
        self.verbose = verbose

    def run(self):
        """
        Start the complete case base reasoning service. It will be running until the user decide to quit. All changes
        and learning can be updated on an xml for being charged in future.
        """
        # Start the loop for solving recipes
        anything_learned, continue_process = False, True
        while continue_process:
            # --------------------------------- INPUT STEP -----------------------------------
            print("Select the ingredients that you want to include in your recipe")
            ingredients = self.request_ingredients_to_user()#['tomato', 'lettuce', 'food']

            # -------------------------------- RETRIEVE STEP -----------------------------------
            base_recipe = self.retrieve(ingredients=ingredients)

            if self.verbose:
                print("\nMost Similar Recipe Found: ")
                print(recipe_to_txt(recipe=base_recipe))

            # --------------------------------- ADAPT STEP -----------------------------------
            adapted_solution = adapt_solution(ingredients=ingredients, base_recipe=base_recipe,
                                              domain_knowledge=self.domain_knowledge,case_library=self.case_library,
                                              verbose=self.verbose)

            # Print adapted solution
            if self.verbose:
                print("\nNew solution has been constructed:")
                print(recipe_to_txt(recipe=adapted_solution))

            # ------------------------------ EVALUATION STEP -----------------------------------
            rate = self.ask_user_for_rating()

            # ------------------------------- LEARNING STEP -----------------------------------
            # Propose the case for be learned
            anything_learned = self.propose_for_learning(adapted_solution=adapted_solution, rate=rate) or anything_learned
            # Review if there is any case within the library which could be forgotten
            anything_learned = self.case_library.forget_step(verbose=self.verbose) or anything_learned

            # -------------------------------- ENDING STEP -----------------------------------
            continue_process = confirm("Would you like to construct more recipes?")

        # If the original database had any update, ask to the user for flushing it on memory
        if anything_learned:
            flush = confirm("Would you like to save the changes for using them in future?")
            if flush:
                self.case_library.flush_database()
        # Bye bye
        print("Thank you! See you next time")

    # ----------------------------- INPUT STEP FUNCTIONS -----------------------------------
    def request_ingredients_to_user(self):
        """
        Ask to the user the input ingredients. The validity of the input will be checked. The spelling mistakes
        will be also checked proposing solutions on the fly. For example: If user says 'tomto', the system will
        recomend the user if he or she wanted to say 'Tomato'. The function also will check that there are no
        repeated ingredients. Empty string is the scaping flag.
        :return: List of str. The ingredients requested from the user
        """

        print("Write one by one the ingredients you would like or press enter (write nothing) for ending")
        ingredients = []
        i = 0
        # While the user still including ingredients (and the amount of ingredients is not unrealistically large)
        while i<MAX_INGREDIENTS:
            # Ask for the ingredient
            user_input = input("Ingredient "+str(i)+": ").lower()
            # Check if the user introduced the scape flag instead of an ingredient
            if user_input == '':
                break
            # If the ingredient is not valid (There is no knowledge about it or it is an spell mistake) propose replacements
            if not self.case_library.check_ingredients_validity(ingredients=[user_input]):
                proposition = self.case_library.propose_ingredient_mistake_solutions(ingredients=[user_input])[0]
                # If user accepts the replacement use this as the new ingredient proposition
                if confirm("Did you want to say "+proposition.title()):
                    user_input = proposition.lower()
                # If do not, ask again for another ingredient
                else:
                    print("Unknown ingredient skipped")
                    continue
            # If the input ingredient repeated skip it
            if user_input in ingredients:
                print("Ingredient repeated. Skipped.")
            # If it is a valid ingredient, add it to the list
            else:
                ingredients.append(user_input)
                i+=1
        # If the user is including an unrealistically large set of ingredients, stop the madness
        else:
            print("Maximum amount of ingredients reached")

        # If the list of ingredients is empty, ask for a new list
        if len(ingredients) == 0:
            print("It is not possible to build an empty recipe.")
            return self.request_ingredients_to_user()

        # Finally, ask for the user final confirmation in order to accept it as the input
        print("This ingredient selection is correct?: "+str([ingredient.title() for ingredient in ingredients])[1:-1]+".")
        if confirm(''):
            return ingredients
        else:
            return self.request_ingredients_to_user()


    # -------------------------------- RETRIEVE STEP FUNCTIONS -------------------------------------
    def retrieve(self, ingredients):
        """
        Ask the database for generating the promissing cases and getting the best one among them
        :param ingredients: List of str. Ingredients requested by the user
        :return: Dictionary. Best recipe found.
        """
        # Return the best case from the database, and if verbose is required also the dissimilarity of the promissing cases
        best_case = self.case_library.retrieve(desired_description=ingredients, return_promising_dissimilarities=self.verbose)
        if self.verbose:
            # Print the count of promising cases and its dissimilarity
            best_case, promising_cases = best_case
            print("Founded "+str(len(promising_cases))+" similar recipes (Promissing Cases). With the following "
                    "dissimilarit"+("y" if len(promising_cases) == 1 else "ies")+": "+str(promising_cases)[1:-1])
        return best_case



    # -------------------------------- EVALUATION STEP FUNCTIONS ----------------------------------
    def ask_user_for_rating(self):
        """
        Ask the user for rating the solution given
        :return: Float between MIN_RATE and MAX_RATE. The rate given by the user for this solution
        """
        # Simulate do-while with an always False first flag
        rate = MIN_RATE-1
        print("How do you rate this solution?")
        # Until the user give a valid response
        while not (MIN_RATE<=rate<=MAX_RATE):
            # Ask for a rating
            response = input("Give a mark (From "+str(MIN_RATE)+" to "+str(MAX_RATE)+"): ")
            # Check its validity for giving which is the error if response is not valid
            try:
                rate = float(response)
                if not (MIN_RATE<=rate<=MAX_RATE):
                    print("Please, give a mark between "+str(MIN_RATE)+" and "++str(MAX_RATE)+".")
            except ValueError:
                print("Please, use a floting point number in range ["+str(MIN_RATE)+", "+str(MAX_RATE)+"]")
        # Return the user response
        return rate

    # --------------------------------- LEARNING STEP FUNCTIONS -----------------------------------
    def take_observation(self, ingredients):
        """
        Used for learning by observation. It ask the user for a better solution than the given by the algorithm. It should
        be called if the user rated the solution with a slow mark (usually less than 3/4 of the maximum rate).
        :param ingredients: List of str. Ingredients which the new recipe includes.
        :return: List of str. The new steps proposed for the user, or an empty list if the user do not propose anything.
        """
        new_steps = []
        # If the user accepts to give a better solution
        if confirm("Would you give a better solution?"):
            # Show the ingredients which the recipe includes as remainder
            print("Ingredients: "+', '.join(ingredients).title())
            # Set the flag to False for simulating a do-while loop
            correct_recipe = False
            # While the result given is not correct (The user have not confirmed its validity)
            while not correct_recipe:
                # Ask the user for give the new steps of the recipe
                print("Please, write the steps one by one. Press Enter for saving the step and press it in an empty step for ending")
                # Initialize the steps information
                new_steps, i = [], 0
                ingredients_unused = ingredients.copy()
                while i<MAX_STEPS:
                    # Ask for the information of this step
                    step = input("Step "+str(i)+": ")
                    # If user uses the scape flag break the bucle
                    if step == '':
                        break
                    # Else include the step
                    else:
                        ingredients_used = find_ingredients_used_by_step(ingredients, [step.lower()], self.domain_knowledge)
                        if len(ingredients_used)>0:
                            verbose_step_matches(ingredients_used, only_one_step=True)
                            ingredients_used = [ingredient for ingredient, reference in ingredients_used[0]]
                            ingredients_unused = ingredients_unused[np.isin(ingredients_unused, ingredients_used) == False]
                            new_steps.append(step)
                            i+=1
                        else:
                            print("All steps must include at least one ingredient")
                # If the recipe is not valid (because it is empty)
                if len (new_steps) == 0 or len(ingredients_unused)>0:
                    if len(new_steps) == 0:
                        print("Every recipe must have one or more steps")
                    elif len(ingredients_unused)>0:
                        print("Every recipe must use all the ingredients. This does not use: "+', '.join(ingredients_unused).title())
                    # Ask to the user if he or she still wanting to take the trouble to give a better solution
                    if not confirm('Would you give a better recipe?'):
                        # If not, show our pain and exit from the function (returning the empty steps list)
                        print('It would have been nice to learn from you. Thank you')
                        return []
                # Else (if the recipe have steps) ask the user if he or she agrees with his or her response
                else:
                    correct_recipe = confirm("These are the correct steps for this recipe?")
        # Return the new steps defined by the user (or an empty list if he or she did not give any better response)
        return new_steps

    def propose_for_learning(self, adapted_solution, rate):
        """
        Check if a solution is enough representative for being learned and, in this case, implements the learning
        by observation if the user gave a low rating for the proposed solution
        :param adapted_solution: Dictionary. Solution adapted by the engine.
        :param rate: Float between MIN_RATE and MAX_RATE. The rate given by the user for this solution
        :return: Boolean. If the solution was really learned by the system or not.
        """

        # If the solution was awful ask to the expert for learning by observation
        if (rate - MIN_RATE) < ((MAX_RATE - MIN_RATE) * (3 / 4)):
            new_solution= self.take_observation(ingredients=adapted_solution[DESCRIPTION])
            # If the user give a new observation consider it as an exceptional case, since the derived solution from
            # the known cases would be a failure.
            if len(new_solution)>0:
                adapted_solution[SOLUTION], adapted_solution[EXCEPTIONAL] = new_solution, True
                adapted_solution[IDENTIFIER] = adapted_solution[DERIVED_FROM] = input("Name your creation: ").lower()
                # Verbose the new recipe created if required
                if self.verbose:
                    print("\nNew recipe learned by observation:")
                    print(recipe_to_txt(recipe=adapted_solution))
        # The solution is only proposed for learning if the rate is higher than 3/4 of maximum rank or
        # if it is an exceptional case (The user give a correct solution for a problem which was not solved by the adaptation)
        if (rate - MIN_RATE) >= ((MAX_RATE - MIN_RATE) * (3 / 4)) or adapted_solution[EXCEPTIONAL]:
            return self.case_library.propose_for_learning(case=adapted_solution,
                                                          verbose=self.verbose)
        # Do not learn from failures if the user did not give a new better solution
        else:
            return False


def confirm(confirm_str):
    """
    Ask the user for confirm or not (Say yes or no) to a given question. Still asking until the user gives a valid answer
    :param confirm_str: str. Question which will be asked to the user
    :return: Boolean. If the user said Yes (True) or No (False) to the question
    """
    # While the user do not give a valid answer
    while True:
        #Ask the question to the user
        confirm = input(confirm_str+" (Yes/No):").lower()
        # If answers Yes return true
        if confirm in ['y', 'yes']:
            return True
        # If answers No return False
        elif confirm in ['n', 'no']:
            return False
        # Else ask again
        else:
            print("Please type only 'Yes' (or 'Y') or 'No' (or 'N')...")


