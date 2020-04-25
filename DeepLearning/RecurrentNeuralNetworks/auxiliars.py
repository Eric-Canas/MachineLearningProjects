import os
import numpy as np
import warnings


def create_directories(model_name):
    """
    Create the directories where save the model, the results and the prediction examples
    :param model_name: name describing the model experimented
    :return:
    Generated paths
    """
    path = os.path.join('results', model_name)
    if not os.path.exists(path=path):
        os.makedirs(path)
    plots_path = os.path.join(path, "Plots")
    if not os.path.exists(path=plots_path):
        os.makedirs(plots_path)
    model_path = os.path.join(path, "Model")
    if not os.path.exists(path=model_path):
        os.makedirs(model_path)
    examples_path = os.path.join(path, "ValidationPredictions")
    if not os.path.exists(path=examples_path):
        os.makedirs(examples_path)
    return plots_path, model_path, examples_path

def to_strokes(strokes_flatten):
    """
    Transform an strokes flatten description to a separated x and y description for easily plot it
    :param strokes_flatten: flatten array of strokes usually used as input for the network
    :return:
    separated x and y description for easily plot it
    """
    positions = list(np.where(strokes_flatten.T[-1] == 2)[0])+[len(strokes_flatten)]
    x = [[strokes_flatten.T[0][i] for i in range(positions[pos-1], positions[pos])]
                                  for pos in range(1, len(positions))]
    y = [[strokes_flatten.T[1][i] for i in range(positions[pos-1], positions[pos])]
                                  for pos in range(1, len(positions))]
    return x,y

def save_model(model, path):
    """
    Save at path, the json description of the model, its weights and a CompleteModel h5 for charging it directly
    :param model: Keras model
    :param path: Path where save it
    """

    with open(os.path.join(path, 'model.json'), 'w') as json_file:
        json_file.write(model.to_json())

    weights_file = os.path.join(path, 'weights' + ".hdf5")
    model.save_weights(weights_file, overwrite=True)

    model.save(filepath=os.path.join(path, 'CompleteModel.h5'), include_optimizer=True)


def name_of_experiment(convolutional_parameters, recurrent_parameters, dense_parameters, num_classes,
                       training_instances, learning_rate, preprocessing, batch_size):

    """
    Translate the experiment parameters to a Natural Language Description
    """
    txt = "Network:\n"+"-"*100+"\n"
    if convolutional_parameters['use']:
        txt+="\tUsing "+str(convolutional_parameters['layers'])+" Convolutional layers with "+\
             str(convolutional_parameters['neurons'])+" filters"
        if convolutional_parameters['dropout_rate'] not in (None, 0):
            txt+=", dropout "+str(convolutional_parameters['dropout_rate'])
        txt+= " and "+convolutional_parameters['activation']+" activation.\n"+"-"*100+"\n"
    txt+= "\tWith "+str(recurrent_parameters['layers'])
    if recurrent_parameters['bidirectional']:
        txt += " Bidirectional"

    txt+= " Reccurrent layers of type " + recurrent_parameters['type']+" with "+str(recurrent_parameters['neurons'])+" neurons"
    if recurrent_parameters['dropout_rate'] not in (None, 0):
        txt += " and dropout " + str(recurrent_parameters['dropout_rate'])
    txt += ".\n"+"-"*100+"\n"

    txt += "\tEnding in " + str(dense_parameters['layers']) + " Dense layers of "+str(dense_parameters['neurons'])+\
            " neurons with " + dense_parameters['activation'] + " activation"
    if dense_parameters['dropout_rate'] not in (None, 0):
        txt += " and dropout " + str(dense_parameters['dropout_rate'])
    txt+= " before last "+str(num_classes)+" softmax neurons.\n"+"-"*100+"\n"

    txt+="Training with "+str(training_instances)+" Instances per class"+" ("+str(num_classes*training_instances)+" examples).\n"+"-"*100+"\n"
    if preprocessing:
        txt+="Using Preprocessing. \n"+"-"*100+"\n"
    txt+="Batch Size: "+str(batch_size)+".\n"+"-"*100+"\n."
    txt += "Learning Rate: " + str(learning_rate) + ".\n" + "-" * 100 + "\n."

    return txt

def short_name_of_experiment(convolutional_parameters, recurrent_parameters, dense_parameters, num_classes,
                             training_instances, learning_rate, preprocessing,batch_size):
    """
    Gets an abreviation of the experiment parameters for use it as a experiment short name
    """

    txt = "Exp-Lr-"+str(learning_rate)
    if convolutional_parameters['use']:
        txt+="-"+str(convolutional_parameters['layers'])+"-Conv-"+str(convolutional_parameters['neurons'])
        if convolutional_parameters['dropout_rate'] not in (None, 0):
            txt+="-dropout-"+str(convolutional_parameters['dropout_rate'])
    txt+= "-type-"+recurrent_parameters['type']+"-"+str(recurrent_parameters['layers'])+"-layers"
    if recurrent_parameters['bidirectional']:
        txt += "-bidirectional"
    txt+="-neurons-"+str(recurrent_parameters['neurons'])
    if recurrent_parameters['dropout_rate'] not in (None, 0):
        txt += "-dropout-" + str(recurrent_parameters['dropout_rate'])
    txt += "-Denses-" + str(dense_parameters['layers'])+"-neurons-"+str(dense_parameters['neurons'])
    if dense_parameters['dropout_rate'] not in (None, 0):
        txt += "-dropout-" + str(dense_parameters['dropout_rate'])
    if preprocessing:
        txt+="-with-preprocessing"
    txt+="-instances-"+str(num_classes*training_instances)+"-batch-size-"+str(batch_size)

    return txt

def get_reliable_parameters(parameters):
    """
    Given a dictionary of parameters modify it in order to unwrap abbrevations (layers:2 , neurons:64 -> neurons:[64,64])
    Also checks if some parameters are inconsistent and solve it returning warnings
    :param parameters: Dictionary of parameters
    :return:
    Dictionary ensuring consistency
    """
    layers = parameters['layers']
    neurons = parameters['neurons']
    dropout_rate = parameters['dropout_rate']
    if type(neurons) is int:
        neurons = [neurons for i in range(layers)]
    elif type(neurons) in (list, tuple) and len(neurons) < layers:
        warnings.warn("Parameters with bad configuration. "+str(layers)+" layers required but neurons found -> "+str(neurons))
        neurons = [neurons[i] if i<len(neurons) else neurons[-1] for i in range(layers)]
        warnings.warn("Modified to "+str(neurons))

    if type(dropout_rate) in (int, float):
        dropout_rate = [dropout_rate for i in range(layers)]
    elif type(dropout_rate) in (list, tuple) and len(dropout_rate) < layers:
        warnings.warn("Parameters with bad configuration. "+str(layers)+" layers required but droprates found -> "+str(dropout_rate))
        dropout_rate = [dropout_rate[i] if i<len(dropout_rate) else dropout_rate[-1] for i in range(layers)]
        warnings.warn("Modified to "+str(dropout_rate))

    new_parameter_description = {'layers': layers,
                                 'neurons': neurons,
                                 'dropout_rate':dropout_rate}
    for key, value in parameters.items():
        if key not in new_parameter_description:
            new_parameter_description[key] = value
    return new_parameter_description

