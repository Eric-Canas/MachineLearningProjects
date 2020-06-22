"""
This file is in charge of all the functions related to solution adaptation. This adaptation is mostly a
Natural Language Processing problem, so this file is focused on solving this kind of problems.
"""

import numpy as np
from pattern.en import pluralize
from DomainKnowledgeController import CHILD_POS
import nltk
from CaseLibraryController import DESCRIPTION, SOLUTION

nltk.download('punkt',quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Define the grammar which we will use for finding the predicates relatives to each ingredient
GRAMMAR_CHUNKER = nltk.RegexpParser('''
    NP: {<DT>? <JJ>* <NN>* <NNS>*} 
    P: {<IN>}           
    V: {<V.*>}          
    PP: {<P> <NP>}      
    VP: {<RB>* <V> <POS>? <NP|PP>* <RB>?}
    NPP: {<CC|CD>? <NP> <PP> <FW>?}
    ''')

USEFUL, PARTIALLY_USEFUL, NON_USEFUL = 1,0,-1
USABILITY_CODE = {USEFUL:'USEFUL', PARTIALLY_USEFUL:'PARTIALLY USEFUL', NON_USEFUL: 'NON USEFUL'}

def adapt_solution(ingredients, base_recipe, domain_knowledge, case_library, verbose=True):
    """
    Main step of solution adaptation, this is the parent function which take the ingredients that we want
    on our new solution, the base recipe from which we will adapt it and adapt the recipe
    :param ingredients: List of str. Ingredients which we want to be in the new recipe.
    :param base_recipe: Dictionary representing a recipe. Recipe from which we will inherit our new solution.
    :param domain_knowledge: DomainKnowledge object. Representation of the domain knowledge of the problem.
    :param case_library: CaseLibrary object. Case Library of the problem.
    :param verbose: Boolean. Verbosing or not verbosing the procedure. Default: True
    :return: Recipe dictionary representing the new created recipe
    """
    # Prepare the data for efficient management
    ingredients = np.array(ingredients, dtype='<U128')
    description, solution = base_recipe[DESCRIPTION], base_recipe[SOLUTION]

    # Find for each step which ingredients are refeared and how (For example: [[(Tomato as Tomatoes),(Pain de Mie as Bread)],...])
    steps_correspondece = find_ingredients_used_by_step(ingredients=description, steps=solution, domain_knowledge=domain_knowledge)
    if verbose:
        verbose_step_matches(steps_correspondece)

    # Check which ingredients are already in the description and which are not
    ingredients_in_recipe = np.isin(ingredients, description)
    ingredients_lack = ingredients[ingredients_in_recipe == False]
    # If there is a lack of ingredients find which are the most similar ingredients in the recipe that will substitute them
    # (For example: Case --> We want mayonnaise and chicken, but the recipe is ketchup and chicken: Substitute for Mayonnaise --> Ketchup.

    used_ingredients, substitutes = ingredients.copy(), []
    if len(ingredients_lack) > 0:
        # Do not repeat ingredients
        substitutes = domain_knowledge.get_most_similar_ingredients(ingredients_requested=ingredients_lack,
                            ingredients_available=np.array(description)[np.isin(description,used_ingredients) == False])
        used_ingredients[ingredients_in_recipe == False] = substitutes
        if verbose:
            verbose_substitution_of_ingredients(ingredients_lack=ingredients_lack, substitutes=substitutes)

    # Find how useful is each step (Don't need adaptation, need adaptation or is not profitable for the new recipe
    usability_of_steps = find_usability_of_steps(correspondences=steps_correspondece, ingredients=used_ingredients, substitutes=substitutes)

    if verbose:
        verbose_usability(solution=solution, usability_of_steps=usability_of_steps)

    # Using all the last information, adapt the solution
    adapted_solution = clean_and_adapt_steps(correspondences=steps_correspondece,
                                                  usability_of_steps=usability_of_steps,
                                                  ingredients=ingredients, steps=solution,
                                                  substitutes=(ingredients_lack, substitutes))
    # Get a new identifier for the new recipe created
    new_identifier = case_library.compose_new_identifier(base_identifier=base_recipe["Identifier"],
                                                              new_ingredients=ingredients)
    # Return the new recipe
    return case_library.compose_case_from_parameters(new_identifier, base_recipe["Identifier"], ingredients,
                                                          adapted_solution)


def clean_and_adapt_steps(correspondences, usability_of_steps, ingredients, steps, substitutes):
    """
    With all the information deduced from how the objective recipe matchs with the origin recipe, transform it to the new solution.

    :param correspondences: List of list of tuples of strings. List of ingredients that each step uses, with the information
    about what is the name of the ingredient and how is referred in step text.
    For example: [[(bacon, bacon), (pain de mie, bread)], [(tomato, tomatoes), (mayonnaise, sauce)],...]
    :param usability_of_steps: List of identifiers. List which identifies each step with its usability for the new solution.
        -1: non useful, 0: useful if adapted, 1: completely useful.
    :param ingredients: List of strings. Ingredients required by the user.
    :param steps: List of stings. Steps of the original solution
    :param substitutes: Tuple of lists. Match between the ingredient which we want to include and the ingredient within the
        recipe which will be substituted by our one. For example: ([(Pain de Mie, Chicken), (Pita Bread, Turkey)])
    :return:
        List of str. List of the adapted steps which composes the new solution.
    """
    # Initialize the list which will save the list of solution steps
    adapted_solution_steps = []

    for step, step_correspondences, usability in zip(steps, correspondences, usability_of_steps):
        # If the step is completely profitable write it unmodified on the new solution
        if usability == USEFUL:
            adapted_solution_steps.append(step)
        # If it is only partially useful adapt it before saving
        elif usability == PARTIALLY_USEFUL:
            # Get the transformation vector of the step (Vector which includes how each ingredient must be changed).
            # For example: From ["Bacon", "Tomato", "Ketchup", "Pita bread"] return ["Bacon", "Tomato", "Mayonnaise", "Pain de Mie"].
            transformation = transformation_vectors(ingredients_requested=ingredients, step_ingredients=step_correspondences,
                                                    substitutes_vector=substitutes)
            # Transform the step with this vector and append it to the new solution
            adapted_solution_steps.append(transform_step(step, step_correspondences, transformation))
        # (If it is not useful avoid it)

    # Return the generated set of steps
    return adapted_solution_steps


def transformation_vectors(ingredients_requested, step_ingredients, substitutes_vector):
    """
    Return the transformation vectors for the concrete ingredients of the step
    :param ingredients_requested: List of strings. Ingredients required by the user.
    :param step_ingredients: List of list of tuples of strings. List of ingredients that each step uses, with the information
    about what is the name of the ingredient and how is referred in step text.
    For example: [[(bacon, bacon), (pain de mie, bread)], [(tomato, tomatoes), (mayonnaise, sauce)],...]
    :param substitutes_vector: Tuple of lists. Match between the ingredient which we want to include and the ingredient within the
        recipe which will be substituted by our one. For example: ([(Pain de Mie, Chicken), (Pita Bread, Turkey)])
    :return: Tuple of lists. Transformation vector of the step (Vector which includes how each ingredient must be changed or not).
        For example: From ["Bacon", "Tomato", "Ketchup", "Pita bread"] return ["Bacon", "Tomato", "Mayonnaise", "Pain de Mie"].
    """
    # Initialize the vector of ingredient transformation
    objective_ingredients = []
    # Unpack the substitutes vector ([Ingredients which we will need to change],[Ingredients from which we will take the solution])
    ingredients_lack, substitutes = substitutes_vector
    # Take each step in the recipe and how it is referred (Pain de Mie, Bread)
    for step_ingredient, step_reference in step_ingredients:
        # If this ingredient is in the user requested recipe, save it unmodified
        if step_ingredient in ingredients_requested:
            objective_ingredients.append(step_ingredient)
        # If is in the substitute vector, save the ingredient requested by the user instead
        elif step_ingredient in substitutes:
            objective_ingredients.append(ingredients_lack[substitutes == np.array(step_ingredient)][0])
        # If is neither in requested vector nor substitutes vector, save an empty string for erasing it
        else:
            objective_ingredients.append('')
    # Return the vector of modifications required
    return objective_ingredients


def transform_step(step, step_correspondences, transformation):
    """
    Transform each step from the original recipe to the adapted step of the requested recipe
    :param step: string. Original step of the recipe.
    :param step_correspondences: List of list of tuples of strings. List of ingredients that each step uses, with the information
    about what is the name of the ingredient and how is referred in step text.
    For example: [[(bacon, bacon), (pain de mie, bread)], [(tomato, tomatoes), (mayonnaise, sauce)],...]
    :param transformation: Transformation vector of the step (Vector which includes how each ingredient must be changed or not).
        For example: From ["Bacon", "Tomato", "Ketchup", "Pita bread"] return ["Bacon", "Tomato", "Mayonnaise", "Pain de Mie"].
    :return:
        String. The step adapted to the transformation requested.
    """
    # Taking how each step is referred in the original step and how it must be in the new adapted step
    for (_, ingredient_reference), new_ingredient in zip(step_correspondences, transformation):
        # Using NLP, if ingredient must be erased, delete all its predicate instead of only the ingredient name
        # (for example, delete 'with a bit of mayonnaise' instead of 'mayonnaise')
        if new_ingredient == '':
            ingredient_reference = get_np_chunk_of_word(phrase=step, word=ingredient_reference)
        # Substitute in the string
        step = step.replace(ingredient_reference, new_ingredient)
    # Return the new step
    return step


def find_ingredients_used_by_step(ingredients, steps, domain_knowledge):
    """
    Search which ingredients from the ingredients vector are used in each step. Using Natural Language Processing and
    the tree which represents the domain knowledge, this function will find also ingredients which are referred in a different
    way in the steps. For example: If ingredient is Pain de Mie, it will found even if it is referred in step as 'Bread'
    or 'Breads' instead of 'Pain de Mie'
    :param ingredients: List of str. Base name of the ingredients that the recipe includes
    :param steps: list of str. List of the steps of the recipe.
    :param domain_knowledge: DomainKnowledge object. Object which represent the knowledge of the domain.
    :return:
        List of list of tuples of strings. List of ingredients that each step uses, with the information
        about what is the name of the ingredient and how is referred in step text.
        For example: [[(bacon, bacon), (pain de mie, bread)], [(tomato, tomatoes), (mayonnaise, sauce)],...]
    """
    # Fill ingregredients with its parents, for founding cases where for example: "Pain de Mie", is referred as "Bread".
    ingredients = domain_knowledge.fill_vector_with_all_parents(nodes=ingredients)

    # Fill it also with plurals for catching also plural references to singular nouns ("Breads" instead of "Bread")
    for ingredient_path in ingredients:
        plurals = []
        for word in ingredient_path:
            plural = pluralize(word)
            if plural not in ingredient_path:
                plurals.append(plural)
        ingredient_path.extend(plurals)

    # Return the list of which ingredients are referred in each step and how
    output = []
    for step in steps:
        this_step_ings = []
        for ingredient_path in ingredients:
            for refered_as in ingredient_path:
                if refered_as in step:
                    plural = pluralize(refered_as)
                    if plural in step:
                        refered_as = plural
                    this_step_ings.append((ingredient_path[CHILD_POS], refered_as))
                    break
        output.append(this_step_ings)
    return output

def find_usability_of_steps(correspondences, ingredients, substitutes):
    """
    Detect which is the usability of each step for the new recipe
    :param correspondences: List of list of tuples of strings. List of ingredients that each step uses, with the information
        about what is the name of the ingredient and how is referred in step text.
        For example: [[(bacon, bacon), (pain de mie, bread)], [(tomato, tomatoes), (mayonnaise, sauce)],...]
    :param ingredients: List of str. Ingredients requested by the user
    :return:
     List of identifiers. List which identifies each step with its usability for the new solution.
        -1: non useful, 0: useful if adapted, 1: completely useful.
    """

    usability = []
    # Take for each correspondence only the original ingredients which it uses (not the references)
    original_ingredient_correspondences = [[original for original, reference in step] for step in correspondences]
    # Taking each list of ingredients that an step uses
    for i, step_ings in enumerate(original_ingredient_correspondences):
        # If the step don't use any ingredient of all the ingredients used by the step are also requested by the user
        if len(step_ings) == 0 or (np.all(np.isin(step_ings, ingredients)) and not np.any(np.isin(step_ings, substitutes))):
            usability.append(USEFUL)
        # If not, but at least some ingredients in the step are also requested by the user
        elif not np.any(np.isin(ingredients, step_ings)):
            usability.append(NON_USEFUL)
        # If the step is completely composed by not required ingredients
        else:
            usability.append(PARTIALLY_USEFUL)
    # Return the array of identifiers
    return np.array(usability,dtype=np.int8)

def get_np_chunk_of_word(phrase, word, grammar_chunker=GRAMMAR_CHUNKER, verbose=True):
    """
    Using Natural Language Processing get the complete predicate which is used for referring a word:
    For example: For the word  'Mayonnaise' and the phrase 'smear the bread with a bit of mayonneise',
    return 'with a bit of Mayonnaise'.
    :param phrase: str. The phrase where searching complex predicates. Usually the step.
    :param word: str. The word for which we will search it predicate. Usually an ingredient.
    :param grammar_chunker: NLTK RegexParser. The grammar chunker which includes the grammar defined for
     solving the problem.
    :param verbose: boolean. If verbose or not when a predicate is satisfactory found.
    :return:
        The complete predicate of the searched word or the word itself if not predicates were found around it.
    """
    # Split the phrase in words and tag them ('Noun', 'Verb', 'Adjective'...)
    sentence = nltk.word_tokenize(phrase)
    sentence = nltk.pos_tag(sentence)
    # Build the trees with the phrase predicates
    result = grammar_chunker.parse(sentence)

    # Search on these trees which is the prepositional, verbal or nominal predicate which includes the searched word
    for tree in result:
        if type(tree) is nltk.tree.Tree and tree.label() in ['PP', 'VP', 'NPP', 'NP']:
            tree_words = [w for w, _ in tree.leaves()]
            # If found save the complete predicate
            if word in tree_words:
                word = ' '.join(tree_words)
                # If verbose show to the user that a concrete predicate was found rounding the ingredient
                if verbose:
                    print('Erasable '+str(tree.label())+': '+word)
                break

    # return the word or the predicate if found
    return word

def verbose_usability(solution, usability_of_steps):
    """
    Print the usability matches in an human readable way
    :param solution: List of str. Steps of the original recipe
    :param usability_of_steps: List of ids. Usability of these steps
    """
    print('\nUsability of the original solution:')
    # For each step print the usability translation and the step
    for usability, step in zip(usability_of_steps, solution):
        print(USABILITY_CODE[usability] + " --> " + step)

def verbose_substitution_of_ingredients(ingredients_lack, substitutes):
    """
    Print from which ingredient will be adapted each ingredient which will need adaptation
    :param ingredients_lack: List of str. Ingredients which was missed on the selected recipe
    :param substitutes: List of str. Ingredients from which them will be adapted
    """
    print('\nThis recipe lack the following ingredients: ' +
          str([ingredient.title() for ingredient in ingredients_lack])[1:-1])
    print("They will be adapted from: ")
    for original, adapted_from in zip(ingredients_lack, substitutes):
        print(original.title() + " --> " + adapted_from.title())

def verbose_step_matches(steps_correspondece, only_one_step = False):
    """
    Verbose which detections of ingredients have been done in each step
    :param steps_correspondece: List of list of tuples of strings. List of ingredients that each step uses, with the information
        about what is the name of the ingredient and how is referred in step text.
        For example: [[(bacon, bacon), (pain de mie, bread)], [(tomato, tomatoes), (mayonnaise, sauce)],...]
    """
    if not only_one_step:
        print("The following matches have been found between Description and Solution")
    for i, step in enumerate(steps_correspondece):
        txt = "Step " + str(i) + " uses: " if not only_one_step else "Ingredients used: "
        for ingredient, reference in step:
            txt += ingredient.title()
            if ingredient != reference:
                txt += ' (Refered as ' + reference.title() + ')'
            txt += ', '
        print(txt[:-len(', ')] + '.')
