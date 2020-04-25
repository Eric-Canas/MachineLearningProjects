import numpy as np
from itertools import combinations
from collections import namedtuple

#Named Tuple with the definition of a condition
condition = namedtuple(typename='Condition', field_names='attributes values consequent coverage precision')

class Rules:
    """
    Implementation of the Rules algorithm proposed by D. T. Pham & M. S. Aksoy at
    Expert Systems With Applications, Vol. 8, No. 1, pp. 59-65, 1995.

    RULES: A Simple Rule Extraction System
    """
    def __init__(self, optimize_unclassifiables = False):
        """
        Initalize the object
        :param optimize_unclassifiables: If enabled, unclassifiable instances are erased from the dataset
        during the combinatorial part an re-added at the end for creating specific rules (optimization 1a,
        explained at complementary report). Else, they are not erased but an additional stop condition is
        enabled in order to stop the combinatorial search when only they remains (optimization 1b,
        explained at complementary report).
        """
        self.optimize_unclassifiables = optimize_unclassifiables
        self.ruleset = []
        # Counts the combinations and selectors explored in order to get performance statistics.
        self.combinations_explored = 0
        self.selectors = 0

    def fit(self, X, Y):
        """
        Creates the ruleset for predicting Y in function of X by the RULES method.
        :param X: 2 dimensional ndarray of inputs
        :param Y: 1 dimensional ndarray of expected outputs for each input row in X
        :return:
            Self. The trained object with the ruleset created.
            Each rule contains coverage and precision information
        """

        datalen = len(Y)
        # At first iteration, the combinations will be over only one attribute
        attr_to_combine = 1
        # If optimization 1a is enabled erase the unclassifiable instances
        if self.optimize_unclassifiables:
            X, Y, unclassifiable_X, unclassifiable_Y = self.detect_unclassifiable(X=X,Y=Y)

        # Stop Condition -> There are no more possible combinations? or Are all elements are classified?
        stop_condition = lambda: (attr_to_combine >= X.shape[-1]) or (len(X) == 0)
        while not stop_condition():

            # For each possible combination of attributes formed by attr_to_combine elements
            for combination in combinations(iterable=range(X.shape[1]), r=attr_to_combine):
                # Possible selectors with this combination of attributes (i.e attr-0 -> 'a', attr-0 ->'b')
                selectors = np.unique(X[:, combination], axis=0)

                # For each selector see if there is only one possible Y value
                for selector in selectors:

                    # Find all instances (within the remaining ones) that fits with this selector
                    satisfy_mask = np.all(X[:, combination] == selector, axis=-1)
                    # Find the possible Y values of this instances
                    consequents = np.unique(Y[satisfy_mask])

                    # If there is only one possible Y value
                    if len(consequents) == 1:

                        if not self.is_irrelevant_condition(combination, selector):
                            # Erase the classified instances from the remaining instances
                            unclassified_instances = satisfy_mask == False
                            X, Y = X[unclassified_instances], Y[unclassified_instances]

                            # Create and append the rule to the ruleset: IF combination = selector THEN consequent
                            self.ruleset.append(condition(combination, selector, consequents[0],
                                                            # Coverage
                                                            np.round(np.sum(satisfy_mask)/datalen, 5),
                                                            # If rule is found at this step precision must be 1
                                                            1.))

                            # Check if there is no sense in continue:
                            # (1. No remaining instances or
                            # 2. Optimization 1b enabled and only unclassifiable instances remains).
                            # Optimization 1b not defined at original algorithm.
                            if (len(X) == 0) or (not self.optimize_unclassifiables and
                                                     self.only_unclassifiables_remaining(X, Y)):
                                attr_to_combine = X.shape[1]
                                break
                    # Count the selectors explored
                    self.selectors += 1

                # Count the combinations explored
                self.combinations_explored += 1

                # If there was no sense in continue break the combinatory exploration
                if attr_to_combine >= X.shape[1]:
                    break

            # Increment the quantity of attributes to combine for repeating the searching
            attr_to_combine += 1

        # If optimization 1a was enabled, now (being sure that X and Y are empty) set it with the
        # inconsistent instances that were not possible to classify
        if self.optimize_unclassifiables:
            X, Y = unclassifiable_X, unclassifiable_Y

        if len(X) > 0:
            # Create complete rules for classifying the remaining instances
            self.classify_rest_(X_rem=X, Y_rem=Y, datalen=datalen)

        #Return self, following the scikit-learn standars.
        return self

    def is_irrelevant_condition(self, combination, selector):
        """
        Check the ruleset in order to found irrelevant condition in a potential rule
        :param combination: attributes of the selectors
        :param selector: value of this attributes
        :return:
        Boolean, if it are irrelevant conditions
        """

        for rule in self.ruleset:
            if np.all([rule_attribute==combination[i] and rule_values==selector[i]
                       for i, (rule_attribute, rule_values) in
                        enumerate(zip(rule.attributes, rule.values))]):
                            return True
        return False

    def get_non_classified_mask(self, X):
        """
        Gets the mask of samples that remains unclassified
        :param X: 2 dimensional ndarray of the X samples
        :return:
        1 dimensional ndarray mask of the X samples unclassified
        """
        return self.predict(X) == ''

    def detect_unclassifiable(self, X, Y):
        """
        Split a dataset in consistent and inconsistent data
        :param X: 2 dimensional ndarray of the X samples
        :param Y: 1 dimensional ndarray of his correspondents Y values
        :return:
        X and Y which are classificable, X and Y which are not
        """

        uniques, reverse = np.unique(X, axis=0, return_inverse=True)
        unclassifiable = np.zeros(len(X), dtype=np.bool)
        if len(uniques) < len(X):
            for i in range(len(X)):
                reverse_mask = reverse == i
                if len(np.unique(Y[reverse_mask])) > 1:
                    unclassifiable = np.bitwise_or(unclassifiable, reverse_mask)
        classifiable = unclassifiable == False
        return X[classifiable], Y[classifiable], X[unclassifiable], Y[unclassifiable]

    def only_unclassifiables_remaining(self, X, Y):
        """
        Checks if only unclassifiables remains unclassified with the current ruleset
        :param X: 2 dimensional ndarray of the X samples
        :return:
        Boolean. True if only unclassifiables remains unclassified with the current ruleset
                False elsewhere
        """
        X, _,_,_ = self.detect_unclassifiable(X=X, Y=Y)
        return self.are_all_classified(X=X)

    def classify_rest_(self, X_rem, Y_rem, datalen):
        """
        Completes the ruleset creating a complete rule for each unclassified instance.
        In case of inconsistencies, it generates for each posible combination of X values
        its most probable Y value rule
        :param X_rem: 2 dimensional ndarray of the X samples which remains
         unclassificable after the completing the attribute combinatory.
        :param Y_rem: 1 dimensional ndarray of his correspondents Y values.
        :param datalen: Len of the original dataset, in order to compute the precision.
        :return:
            None. Completes the self.ruleset list.
        """
        # For each possible combination of X values remaining using all attributes
        for X_value in np.unique(X_rem, axis=0):
            # Founds the Y values for this elements combinations
            satisfy_mask = np.all(X_rem == X_value, axis=1)
            Y_values, support_all = np.unique(Y_rem[satisfy_mask], return_counts=True)
            # Decide which is the most probable Y value for this X combination
            most_probable_Y, support = Y_values[np.argmax(support_all)], support_all.max()

            # Create the rule using all the attributes and assigning the most probable Y value
            self.ruleset.append(condition(tuple(range(X_rem.shape[1])), X_value, most_probable_Y,
                                          np.round(support / datalen, 5),
                                          np.round(support/ np.sum(support_all),3)))

            # Remove this values from the unclassified instances
            satisfy_mask = satisfy_mask == False
            X_rem, Y_rem = X_rem[satisfy_mask], Y_rem[satisfy_mask]

    def are_all_classified(self, X):
        """
        Return True if are elements of X are classified with the ruleset self.ruleset
        :param X: 2 dimensional ndarray of samples to predict
        :return:
        True if are all instances of X classifiable with the ruleset self.ruleset, False elsewhere.
        """
        if len(self.ruleset) == 0:
            return False
        else:
            return not np.any(self.predict(X) == '')

    def predict(self, X, less_specific_priority = True):
        """
        Predicts the expected values of Y for the array X.
        :param X: 2 dimensional ndarray of samples to predict
        :param less_specific_priority: If true, the less specific rules has priority
        (i.e "If 0 is 'A' then 'B'" prevails over "If 0 is 'A' and 1 is 'C' then 'A'").
        Else the most specific prevails.
        Original algorithm proposes to do it always in True mode.
        :return:
            1 dimensional ndarray with the predictions of X.
        """
        Y = np.zeros(len(X), dtype='<U128')

        # If wants to apply the less specific priority rule, reverse the ruleset.
        ruleset = self.ruleset
        if less_specific_priority:
            ruleset = self.ruleset[::-1]

        # For each rule classify, overriding if a most priority rule appears after.
        for rule in ruleset:
            Y[np.all(X[:, rule.attributes] == rule.values, axis=1)] = rule.consequent
        return Y

    def print_rules(self, names, less_specific_first = True):
        """
        Prints the ruleset in a readable format
        :param names: list with the names of the attributes
        :param less_specific_first: Orders with less specific rules first
        :return:
            None
        """

        # If wants to apply the less specific priority rule, reverse the ruleset.
        ruleset = self.ruleset
        if not less_specific_first:
            ruleset = ruleset[::-1]

        # For each rule print it
        for i,rule in enumerate(ruleset):
            print("RULE "+str(i)+". Coverage: "+str(rule.coverage)+". Precision: "+str(rule.precision))
            print(get_rule(rule=rule, names=names))

def get_rule(rule, names):
    """
    Gets a string with a rule written in a readable format
    :param rule: rule to write in internal ruleset format
    :param names: list with the names of the attributes
    :return:
        string with the rule in a readable format
    """
    # IF
    rule_txt = "IF "

    # 'Attribute i' -> 'Value_k' AND 'Attribute j' -> 'Value_l'...
    for i in range(len(rule.attributes)):
        rule_txt += names[rule.attributes[i]] + ' -> '+rule.values[i]
        if i<len(rule.attributes)-1:
            rule_txt += ' AND '

    # THEN 'Class' --> 'Predicion'
    rule_txt += ' THEN ' + names[-1] + ' --> ' + rule.consequent

    return rule_txt
