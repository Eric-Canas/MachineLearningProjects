from Reader import get_dataset_k_folds
import numpy as np
from Forest import Forest
import os

experiments = {'random_forest':
                                {'NT':[1,10,25,50,75,100,250,500],
                                 'F':[lambda M: 1,
                                      lambda M: 3,
                                      lambda M: int(np.log2(M+1)),
                                      lambda M: int(np.sqrt(M))]},
               'decision_forest':
                                {'NT':[1,10,25,50,75,100,250,500],
                                 'F':[lambda M: lambda: M//4,
                                      lambda M: lambda: M//2,
                                      lambda M: lambda:(3*M)//4,
                                      lambda M: lambda: np.random.randint(low=1, high=M)]}
               }

def analyze(file,folds=10):
    """
    Executes the main analysis of a Dataset using the algorithm.
    Reading the dataset from 'file' csv, executes a k-folds Accuracy analyisis,
    with k=folds.
    :param file: File where reading the dataset
    :param folds: K of the k-folds analysis. If -1 then a Leave One Out
    analysis will be performed.
    :return:
    Prints the mean validation accuracy.
    """
    folder = os.path.join('Results', 'dataset-'+file[:-len('.csv')])


    if not os.path.isdir(folder):
        os.makedirs(folder)
    for forest_type, params in experiments.items():
        forest_folder = os.path.join(folder,forest_type)
        if not os.path.isdir(forest_folder):
            os.makedirs(forest_folder)
        acc_matrix = []
        for NT in params['NT']:
            acc_row = [NT]
            for F in params['F']:
                # For each fold
                accuracy = []
                attributes_relevance = []
                for i, (X_train, Y_train, X_val, Y_val, names) in enumerate(get_dataset_k_folds(file=file, k=folds)):
                    M = X_train.shape[-1]
                    # Fit the forest
                    forest = Forest(F=F(M), NT=NT,forest_type=forest_type).fit(X=X_train, Y=Y_train, X_names=names[:-1])
                    # Predict values for the validation fold
                    predicted = forest.predict(X=X_val, X_names = names[:-1])
                    # Extract the attribute relevance on this forest
                    attributes_relevance.append(forest.attributes_relevance())
                    # Saves the accuracy
                    accuracy.append(100 * np.sum(Y_val == predicted) / len(Y_val))
                    # Verbose each 20 folds
                    if i % 20 == 0 and i>0:
                        print(str(i)+ " Folds Done")
                accuracy = np.round(np.mean(accuracy),1)
                print(str(i) + " Folds Executed")
                # Print Results
                print("-"*50)
                print("Accuracy: " + str(accuracy) + "%")
                acc_row.append(accuracy)
                f_description = str(F(M)) if forest_type == 'random_forest' else str(F(M)())
                forest.save_attributes_relevance_plot(file=os.path.join(forest_folder, 'F-' + f_description + '-NT-' + str(NT)),
                                                      title=forest_type.replace('_',' ').title()+' - Feature Relevance')
            acc_matrix.append(acc_row)
        print(forest_type)
        with open(os.path.join(forest_folder, 'Acc Matrix '+str(folds)+'-folds.txt'),'w') as f:
            f.write(str(np.array(acc_matrix)))

