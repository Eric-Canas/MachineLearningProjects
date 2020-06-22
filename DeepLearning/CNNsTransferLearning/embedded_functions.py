import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

INPUT = 0
OUTPUT = 1
FILE_NAME = 'features'
LABELS_FILE_NAME = 'labels'
PLOT_MARKERS = ['o','.', 'v', '^', '<', '>', 's', 'p', 'P', 'X', 'D', '1', '2', '3', '4']

def get_embedded_from_dataset(model, dataset, standardize_output=False, reduce_operation=np.mean,
                              embedded_datasets_path=os.path.join('..','Datasets','food-101','embeddeds'),
                              get_hot_encoded=True):
    dataset, dataset_description = dataset
    is_calulated, standard_folder, non_standard_folder = is_calculated(model_name = model.name,
                                                                       dataset_description=dataset_description,
                                                                       path=embedded_datasets_path,
                                                                       reduce_operation=reduce_operation)

    if not is_calulated:
        if reduce_operation is not None:
            features = np.concatenate([reduce_operation(model.predict(dataset[batch][INPUT]),axis=(-3,-2))
                                        for batch in range(len(dataset))])
        else:
            batch_size = len(dataset[0][INPUT])
            features = np.concatenate([model.predict(dataset[batch][INPUT]).reshape(batch_size, -1)
                                       for batch in range(len(dataset)-1)])

        #np.save(file=os.path.join(non_standard_folder, FILE_NAME), arr=features)
        #np.save(file=os.path.join(non_standard_folder, LABELS_FILE_NAME), arr=Y)
        if standardize_output:
            features -= np.mean(features, axis=0)
            std = np.std(features, axis=0)
            features[:, std!=0.] /= std[std!=0.]
        Y = dataset.labels[:len(features)]
        #np.save(file=os.path.join(standard_folder, FILE_NAME),arr=features)
        #np.save(file=os.path.join(standard_folder, LABELS_FILE_NAME),arr=Y)
    else:
        if standardize_output:
            features = np.load(file=os.path.join(standard_folder, FILE_NAME+'.npy'))
            Y = np.load(file=os.path.join(standard_folder, LABELS_FILE_NAME+'.npy'))
        else:
            features = np.load(file=os.path.join(non_standard_folder, FILE_NAME + '.npy'))
            Y = np.load(file=os.path.join(non_standard_folder, LABELS_FILE_NAME + '.npy'))
    if get_hot_encoded:
        Y = to_categorical(Y)
    return features, Y

def is_calculated(model_name, dataset_description, path, reduce_operation):
    reduce_operation = 'Mean' if reduce_operation is not None else 'Flatten'
    standard_folder = os.path.join(path, model_name,reduce_operation, dataset_description, 'standardized')
    non_standard_folder = os.path.join(path, model_name,reduce_operation, dataset_description, 'not-standardized')
    exists_folder = os.path.isdir(standard_folder) and os.path.isdir(non_standard_folder)
    if not exists_folder:
        os.makedirs(standard_folder)
        os.makedirs(non_standard_folder)
        return False, standard_folder, non_standard_folder
    else:
        return len(os.listdir(non_standard_folder))+ len(os.listdir(standard_folder)) == 4, standard_folder, non_standard_folder


def plot_embedded_space(train_dataset, val_dataset, test_dataset, embedded_space_dir):
    validation_dir, train_dir, test_dir = [os.path.join(embedded_space_dir, dir_) for dir_ in ('validation', 'train', 'test')]
    [os.makedirs(path) for path in (validation_dir, train_dir, test_dir) if not os.path.isdir(path)]
    X_train, Y_train = train_dataset
    X_val, Y_val = val_dataset
    X_test, Y_test = test_dataset
    plot_predictors_confusion_matrices(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test,
                                       Y_test=Y_test, validation_dir=validation_dir, train_dir=train_dir, test_dir=test_dir)

    plot_lda(X_train,Y_train, X_val, Y_val, embedded_space_dir=validation_dir)
    plot_TSNE(X_val, Y_val, embedded_space_dir=validation_dir)

def plot_predictors_confusion_matrices(X_train, Y_train, X_val, Y_val, X_test, Y_test, validation_dir, train_dir, test_dir):
    clfs = tuple([(MLPClassifier(neurons, max_iter=250, batch_size=256, learning_rate='adaptive', learning_rate_init=0.0075/(neurons/1024), verbose=True), 'Overfitted MLP '+str(neurons)+' Neurons at Hidden')
                                                      for neurons in [128, 256, 512, 1024, 2048, 4096]]) + \
           tuple([(MLPClassifier(neurons, max_iter=250, batch_size=256, learning_rate='adaptive',
                                 learning_rate_init=0.0075/(neurons/1024), verbose=True,early_stopping=True,validation_fraction=0.2),
                   'Early Stopping MLP ' + str(neurons) + ' Neurons at Hidden')
                  for neurons in [128, 256, 512, 1024, 2048, 4096]]) + \
           ((RandomForestClassifier(min_samples_split=5, verbose=True), 'Random Forest'),
               (svm.LinearSVC(max_iter=100, verbose=True), 'Linear SVC'))
            #(KNeighborsClassifier(n_neighbors=1),'1 Nearest Neighbors'),
            #(KNeighborsClassifier(n_neighbors=3),'3 Nearest Neighbors'))
    for clf, name in clfs:
        if not was_experimented(name, train_dir, validation_dir, test_dir):
            clf.fit(X=X_train, y=Y_train)
            for X, Y, dir in ((X_train, Y_train, train_dir), (X_val, Y_val, validation_dir), (X_test, Y_test, test_dir)):
                clf_score = str(np.round(clf.score(X=X, y=Y), decimals=3))
                clf_pred = clf.predict(X)
                plot_confusion_matrix(y_true=Y, y_pred=clf_pred, embedded_space_dir=dir,
                                      title=name+" Confusion Matrix - Acc - " + clf_score)

def was_experimented(name, train_dir, validation_dir, test_dir):
    return len([file for dir_ in (train_dir, validation_dir, test_dir) for file in os.listdir(dir_) if name in file]) >= 3


def plot_confusion_matrix(y_true, y_pred, embedded_space_dir, title):
    plt.matshow(confusion_matrix(y_true, y_pred), cmap='Reds')
    plt.colorbar()
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(embedded_space_dir, title+'.png'))
    plt.close()


def plot_lda(X_train, Y_train, X_val, Y_val, embedded_space_dir, classes=101):
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit(X_train, y=Y_train).transform(X_val)
    for label in np.unique(Y_val):
        plt.plot(X_lda[Y_val == label][:, 0], X_lda[Y_val == label][:, 1], np.random.choice(PLOT_MARKERS), c=np.random.rand(3),
                 alpha=0.75)
    lda_score = LinearDiscriminantAnalysis(n_components=classes).fit(X=X_train, y=Y_train).score(X=X_val,y=Y_val)
    title = '2 dims LDA - '+str(classes)+' dims Acc - '+str(np.round(lda_score, decimals=3))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(embedded_space_dir, title+'.png'))
    plt.close()

def plot_TSNE(X_val, Y_val, embedded_space_dir, classes=101):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_val)
    for label in np.unique(Y_val):
        plt.plot(X_tsne[Y_val == label][:, 0], X_tsne[Y_val == label][:, 1], np.random.choice(PLOT_MARKERS), c=np.random.rand(3),
                 alpha=0.75)
    title = 'TSNE 2 dims - '+str(classes)+' dims'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(embedded_space_dir, title+'.png'))
    plt.close()
