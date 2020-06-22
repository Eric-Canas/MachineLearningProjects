import numpy as np

COUNTS = 1

class C45:
    """
    Implementation of the C4.5 Classifier
    """
    def __init__(self):
        """
        Initalize the object
        """
        self.is_leaf = False
        self.prediction = None
        self.branches = {}
        self.attribute = None
        self.gains_ratio=None
        self.confidence = 1.

    def fit(self, X, Y, X_names, F = None):
        """
        Creates the C4.5 model for predicting Y in function of X.
        :param X: 2 dimensional ndarray of inputs
        :param Y: 1 dimensional ndarray of expected outputs for each input row in X
        :param X_names: Name of each attribute from X
        :param F: Amount of attributes from total to use in each node (If None, all attributes).
                F should be only used when C4.5 Tree is part of a Random Forest.
        :return:
            Self. The trained object.
        """

        # Pre-computes h_x in order to not repeat its calculation inside.
        h_x = get_split_info(Y)

        # Save as prediction the most common Y. If it is not a leaf it will be only used when new value appears at test
        labels, counts = np.unique(Y, return_counts=True)
        self.prediction = labels[np.argmax(counts)]
        self.confidence = np.max(counts) / len(Y)

        # If entropy of Y is 0 is because it is a leaf (Only one label at Y). If no more X_names label do not have confidence 1.
        if np.isclose(h_x,0,rtol=1e-8) or len(X_names) == 0:
            self.is_leaf = True
            self.gains_ratio = 1.
        else:
            # When F is not None (Random Forest) selects a subsets of attributes to compute with
            attributes_to_use = np.random.choice(range(len(X_names)), size=min(F, len(X_names)), replace=False) \
                                                if F is not None else range(len(X_names))

            # ------------------------- DECIDING THE BEST ATTRIBUTE TO SPLIT ---------------------------
            # Gets each valid attribute Split Info.
            gains_ratio = [(get_gain_ratio(attribute=X[...,att], Y=Y, h_x=h_x) if att in attributes_to_use else -1)
                                                                                for att in range(X.shape[-1])]
            # Decide the attribute with best gains ratio for this node
            attribute = np.argmax(gains_ratio)
            # Saves the gain ratio of the best for future statistics
            self.gains_ratio = gains_ratio[attribute]
            # Saves the attribute name for predicting
            self.attribute = X_names[attribute]

            # --------------------------------- GENERATING BRANCHES -------------------------------------
            # Prepares the data for an efficient future selection:
            values, idx = np.unique(X[...,attribute], return_inverse=True)
            # Set the datasets which the following subtrees will use
            new_X, new_names = np.delete(X, attribute, axis=-1), np.delete(X_names, attribute)

            # Generate the new branches (In DFS) it makes it through the reverse idx because int comparisons are faster than str
            for i, value in enumerate(values):
                new_data_mask = idx == i
                self.branches[value] = C45().fit(X=new_X[new_data_mask], Y=Y[new_data_mask], X_names=new_names, F=F)

        return self

    def predict(self, X, names):
        """
        Predicts the expected values of Y for the array X.
        :param X: 2 dimensional ndarray of samples to predict
        :param names: 1 dimensional ndarray with the names of each attribute in X
        :return:
            1 dimensional ndarray with the predictions of X.
        """
        def predict_recursive(node, instance, names):
            """
            Used internally to predict instances in a recursive way, not callable for the user.
            :param node: current tree of the tree
            :param instance: the instance X without the already analyzed attributes
            :return:
                Prediction for the initial instance
            """
            # If node is a leaf returns the prediction
            if node.is_leaf:
                return node.prediction
            else:
                attribute = names[node.attribute]
                # If is not a leaf it can have a branch or not (if this combination was not in the train set)
                if instance[attribute] in node.branches:
                    new_node = node.branches[instance[attribute]]
                else:
                    return node.prediction
                return predict_recursive(node=new_node, instance=instance, names = names)

        names_dict = {name:i for i,name in enumerate(names)}
        # Predict will only call predict_recursive for each instance, setting its current node as the complete tree
        return np.array([predict_recursive(node=self, instance=x, names=names_dict) for x in X])

    def get_attributes(self):
        """
        Get the set of attributes that tree uses for predictions, exploring the tree in DFS way.
        :return:
        1d array of unique str attributes
        """
        def get_attributes_recursive(node):
            """
            Returns each node attribute in a recursive way
            :param node: node of the tree from which extracting the attribute
            :return:
                list of non unique str attributes
            """

            attributes = []
            # If node is not leaf get its attribute and all the attributes of its sons
            if not node.is_leaf:
                # Save its attribute
                attributes.append(node.attribute)
                # Extend with all its sons attributes
                for son in node.branches.values():
                    attributes.extend(get_attributes_recursive(node=son))

            return attributes

        return np.unique(get_attributes_recursive(node=self))


def get_gain_ratio(attribute, Y, h_x = None):
    """
    Returns the gain ratio of an attribute (Gain(X,A)/SplitInfo(X))
    :param attribute: 1 dimensional
    :param h_x: float. Precalculated entropy of the objective class,
    if None it is calculated. (Introduced by efficiency reasons)
    :return:
    """
    # ------------------------- SPLIT INFO -------------------------------
    # Calculates the split infor taking the correspondencies for reuse it more efficiently
    _,correspondencies, p_x = np.unique(attribute, return_counts=True, return_inverse=True)
    #If all instances have the same value split info will be 0, so return it directly (it will prevent uni-branches)
    if len(p_x) <= 1:
        return 0.
    p_x = p_x / len(attribute)
    # Calculates the SplitInfo(X) (Entropy)
    split_info_x = -np.sum((p_x * np.log2(p_x)))

    # ---------------------- INFORMATION GAIN -----------------------------
    if h_x is None:
        h_x = get_split_info(Y)
    # Calculates the gain
    gain = h_x - np.sum([get_split_info(Y[correspondencies==i])for i in range(len(p_x))]*p_x)

    # ------------------------- GAIN RATIO --------------------------------
    gain_ratio = gain/split_info_x
    return gain_ratio

def get_split_info(attribute):
    """
    Returns the split info (Entropy) of an attribute
    :param attribute: 1 dimensional array. Attribute where extract its split info
    :return:
    float. Split Info of the attribute
    """
    # Gets the frequency of each value (P(x))
    p_x = np.unique(attribute, return_counts=True)[COUNTS]/len(attribute)
    # Calculates the Split Info (Entropy)
    split_info = -np.sum((p_x * np.log2(p_x)))
    return split_info
