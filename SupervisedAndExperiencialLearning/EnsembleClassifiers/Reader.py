import numpy as np
import os

def get_dataset_k_folds(file, root='Data', objective_pos=None, k=10, print_class_distribution=True):
    """
    Reads a dataset in format CSV in which the first row are the names of the columns
    and yields it in an iterable k-fold way. In training set returned, it erases the
    missing values.
    :param file: Name of the file containing the dataset (i.e: example.csv)
    :param root: name of the root directory where file is contained
    :param objective_pos: Position of the objective variable. If None, the position
    will be the one named as class by first row.
    :param validation_split: Quantity of validation data to generate
    :param k: k-folds to return. If -1 Leave One Out mode selected
    :return:
    X_train, Y_train, X_val, Y_val and names ndarrays representing each one of
    the k dataset folds.
    """
    # Charges the dataset
    dataset = np.genfromtxt(os.path.join(root, file), delimiter=',',skip_header=True, dtype=np.str)
    #If K is -1 then Leave One Out mode
    if k==-1:
        k = len(dataset)
    else:
        k = min(k, len(dataset))
    #Shuffle the dataset with a seed, in order to ensure that all experiments will be fair
    np.random.seed(0)
    np.random.shuffle(dataset)

    # Save the names contained at first row of the csv
    with open(os.path.join(root, file)) as f:
        names = f.readline()[:-len('\n')].split(',')

    # Decide the position of the objective variable
    if objective_pos is None:
        objective_pos = names.index('class')
    elif objective_pos < 0:
        objective_pos = dataset.shape[-1]+objective_pos

    if print_class_distribution:
        print("Class Distribution:")
        values, count = np.unique(dataset[:, objective_pos], return_counts=True)
        for i in range(len(values)):
            print("\t"+values[i].title()+" -> "+str(np.round(100*(count[i]/np.sum(count)),decimals=2))+"%")
        print("-"*50)
    # Put the class name at last position
    names.append(names.pop(objective_pos))
    names = np.array(names)
    # Specify the positions of the attributes which are objective variables and which are not
    non_objective_positions = [i for i in range(dataset.shape[-1]) if i != objective_pos]
    # Length of each validation fold
    k_len = int(len(dataset)/k)

    # Yields one fold at each iteration
    for i in range(k):
        # Gets validation and train indexes
        validation_idx = range(k_len*i,k_len*(i+1))
        train_idx = [i for i in range(len(dataset)) if i not in validation_idx]
        # Split the dataset into train fold and validation fold
        train_fold, val_fold = dataset[train_idx], dataset[validation_idx]
        # Erase the rows containing missing values in train
        train_fold = train_fold[np.any(train_fold == '?', axis=1) == False]

        # Yields the fold i
        yield train_fold[..., non_objective_positions], train_fold[..., objective_pos],\
              val_fold[...,non_objective_positions], val_fold[...,objective_pos], names

def bootstrap_data(X_data, Y_data):
    """
    Returns a bootstrapped divided dataset with an average percentage of X
    :param X_data: 2d ndarray. Original X data
    :param Y_data: 1d ndarray. Objective attribute (Y data)
    :param N: Quantity of bootstraped splits to return
    :param uniques_percentage: Average percentage of unique data in sets
    :return:
        Set of splitted bootstrapped datasets to return.
    """
    # Fuse X and Y for easir internal management
    data = np.concatenate([X_data, Y_data[..., None]], axis=-1)
    idx = np.random.choice(range(len(data)), size=len(data),replace=True)
    return data[idx,:-1], data[idx,-1]
