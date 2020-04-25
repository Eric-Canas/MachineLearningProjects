import matplotlib
from matplotlib import pyplot as plt
import os
import numpy as np
from auxiliars import to_strokes
from sklearn.metrics import classification_report, confusion_matrix

round_value = 4

def plot(history, path, label, xlabel='Epochs', round_value=round_value):
    """
    Plots the curves of a determined history (Loss, Accuracy...) of the keras model history
    :param history: model.history of keras model
    :param path: path where saving the results
    :param label: key of the history for saving ('loss', 'accuracy', 'top_3_accuracy'...)
    :param xlabel: label for the X axis
    :param round_value: how many decimals used for the floats
    :return:
    Save a plot image at the given path
    """
    matplotlib.use('Agg')
    over_train, over_val = history.history[label], history.history['val_' + label]

    plt.plot(over_train)
    plt.plot(over_val)
    plt.title('Model '+ label.title())
    plt.ylabel(label.title())
    plt.xlabel(xlabel)
    plt.legend(['Training', 'Validation'], loc='upper left')

    name = label.title() + '. Tr-' + str(np.round(over_train[-1],round_value)) + '. Val-' + str(np.round(over_val[-1],round_value)) + '.png'

    plt.savefig(os.path.join(path, name))
    plt.close()

def save_prediction_examples(path, model, dataset, predictions_to_save = 10):
    """
    Save predictions_to_save examples of the predictions given by the model for the given dataset containing
    the  QuickDrawDataset instances.
    :param path: Path where saving this examples
    :param model: Keras model generated
    :param dataset: Dataset where taking the examples
    :param predictions_to_save: How many predictions examples will be saved
    :return:
     Save the images of the evaluation examples at the given path
    """
    matplotlib.use('Agg')
    x, y = dataset.X[:predictions_to_save], dataset.Y[:predictions_to_save]
    predictions_to_save = np.array([model.predict(instance[None,...])[0] for instance in x])
    top_3_predictions = np.argpartition(-predictions_to_save, kth=3, axis=-1)[:, :3]
    predictions = np.argmax(predictions_to_save, axis=-1)
    for i,(y_pred, top_3_y_pred, y_test) in enumerate(zip(predictions, top_3_predictions, y)):
        title, suptitle, plot_name = generate_tittles_and_name(dataset, i, predictions_to_save, top_3_y_pred, y_pred, y_test)

        strokes = to_strokes(strokes_flatten=x[i])

        plt.gca().invert_yaxis()
        for stroke_x, stroke_y in zip(strokes[0], strokes[1]):
            plt.plot(stroke_x, stroke_y)
        plt.legend(['Stroke '+str(i+1) for i in range(len(strokes[0]))], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        plt.xlabel(suptitle)
        try:
            plt.savefig(os.path.join(path,plot_name), bbox_inches = "tight")
        except FileNotFoundError as err:
            print(err)
        plt.close()


def generate_tittles_and_name(dataset, sample_idx, predictions_to_save, top_3_y_pred, y_pred, y_test):
    """
    Auxiliar function for save_prediction_examples which generates the tittles of the plots
    """
    percentage = round(predictions_to_save[sample_idx][y_pred] * 100, ndigits=2)
    top_3_percentage = np.round(predictions_to_save[sample_idx][top_3_y_pred] * 100, decimals=2)
    y_pred_class, y_test_class, top_3_classes = dataset.classes[y_pred], dataset.classes[y_test], dataset.classes[
        top_3_y_pred]
    top_3_indexes = np.argsort(-top_3_percentage)
    top_3_percentage, top_3_classes = top_3_percentage[top_3_indexes], top_3_classes[top_3_indexes]
    if y_pred == y_test:
        tittle = plot_name = 'Correct'
    elif y_test in top_3_y_pred:
        tittle = plot_name = 'InTop3'
    else:
        tittle = plot_name = 'Wrong'
    plot_name += '-Example ' + str(sample_idx) + ' - ' + y_test_class + '.png'
    tittle += ': Is ' + y_test_class + '. Predicted ' + y_pred_class + ' (' + str(percentage) + '%)'
    suptittle = 'Top 3 Predictions: '
    for label, percentage in zip(top_3_classes, top_3_percentage):
        if label == y_test_class:
            suptittle+= '['
        suptittle += label + '(' + str(percentage) + '%)'
        if label == y_test_class:
            suptittle += ']'
        suptittle += ' - '
    suptittle = suptittle[:-len(' - ')]

    return tittle, suptittle, plot_name


def evaluate_model(model, path, dataset, split, target_names, round_value = round_value):
    """
    For the given model and Dataset generate its Evaluation txt and its Confusion Matrix as png
    :param model: Keras model to analyze
    :param path: Path where saving the results
    :param dataset: Dataset for evaluating
    :param split: Description of the split that dataset represents ('training', 'validation' or 'test')
    :param target_names: Name of the classes
    :param round_value: How many decimals are used for the floating point numbers
    :return:
    Save the Evaluation txt and the Confusion Matrix as png
    """
    results = model.evaluate_generator(generator=dataset, steps=len(dataset), verbose=0)

    with open(os.path.join(path, split + '-' + str(np.round(results[1], round_value)) + '-summary.txt'), 'w') as summary:

        summary.write('### SUMMARY ###\n\n')
        for result, parameter in zip(results, ('Loss', 'Accuracy', 'Top 3 Accuracy')):
            summary.write(parameter+": " + str(np.round(result, round_value)) + '\n')
        summary.write('\n\n')

        # Confusion matrix
        y_pred = np.argmax(model.predict_generator(dataset, steps=len(dataset)), axis=1)
        y_test = dataset.Y[:len(y_pred)]

        summary.write('Analysis of results:\n')
        summary.write(classification_report(y_test, y_pred, target_names=target_names))

    plt.matshow(confusion_matrix(y_test, y_pred),cmap='Reds')
    plt.colorbar()
    plt.title(split.title()+" Confusion Matrix. Acc: "+str(np.round(results[1], round_value)))
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(path,split.title() + " Confusion Matrix. Acc-" + str(np.round(results[1], round_value))+'.png'))
    plt.close()