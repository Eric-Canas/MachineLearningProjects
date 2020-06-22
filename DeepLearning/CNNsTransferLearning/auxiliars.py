import os
import numpy as np
import warnings
import keras
from keras.models import load_model
#https://github.com/matterport/Mask_RCNN/issues/1108
IMPLEMENTED_NETS = ['MobileNet', 'vgg16-hybrid1365.h5','ResNet101', 'ResNet152', 'Xception', 'NasNetLarge', 'DenseNet201', 'VGG16', 'VGG19']
models_path = 'models'

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
    fc_plots_path = os.path.join(path, "FullyConnectedPartPlots")
    if not os.path.exists(path=fc_plots_path):
        os.makedirs(fc_plots_path)
    complete_plots_path = os.path.join(path, "CompleteNetworkPlots")
    if not os.path.exists(path=complete_plots_path):
        os.makedirs(complete_plots_path)
    model_path = os.path.join(path, "Model")
    if not os.path.exists(path=model_path):
        os.makedirs(model_path)
    embedded_space_path = os.path.join(path, "EmbeddedSpace")
    if not os.path.exists(path=embedded_space_path):
        os.makedirs(embedded_space_path)
    return complete_plots_path, fc_plots_path, model_path, embedded_space_path

def save_model(model, path):
    """
    Save at path, the json description of the model, its weights and a CompleteModel h5 for charging it directly
    :param model: Keras model
    :param path: Path where save it
    """

    with open(os.path.join(path, 'model.json'), 'w') as json_file:
        json_file.write(model.to_json())

    weights_file = os.path.join(path, 'weights' + ".h5")
    model.save_weights(weights_file, overwrite=True)

    model.save(filepath=os.path.join(path, 'CompleteModel.h5'), include_optimizer=True)

def short_name_of_experiment(base_model_name, dense_parameters, learning_rate, preprocessing, trained_embedded=False):
    """
    Gets an abreviation of the experiment parameters for use it as a experiment short name
    """

    txt = "Exp-"+base_model_name+"-Lr-"+str(learning_rate)
    txt += "-Denses-" + str(dense_parameters['layers'])+"-neurons-"+str(dense_parameters['neurons'])
    if dense_parameters['dropout_rate'] not in (None, 0):
        txt += "-dropout-" + str(dense_parameters['dropout_rate'])
    if preprocessing:
        txt+="-with-preprocessing"
    if trained_embedded:
        txt+="-train-with-embedded"
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

def get_model(img_shape = (224, 224), model_name='MobileNet', weights='imagenet'):
    path = os.path.join(models_path, model_name.lower()+'.h5')
    print(path)
    if model_name.lower() == 'mobilenet':
        if os.path.exists(path):
            model = load_model(path)
        else:
            model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=img_shape,
                                                                include_top=False,
                                                                # We don't need the classification layer.
                                                                weights=weights)
            model.save(filepath=path, include_optimizer=True)
            print(model_name + ' saved')
        preprocessing = keras.applications.mobilenet_v2.preprocess_input
    elif model_name.lower() == 'xception':
        if os.path.exists(path):
            model = load_model(path)
        else:
            model = keras.applications.xception.Xception(input_shape=img_shape,
                                                                include_top=False,
                                                                # We don't need the classification layer.
                                                                weights=weights)
            model.save(filepath=path, include_optimizer=True)
            print(model_name + ' saved')
        preprocessing = keras.applications.xception.preprocess_input
    elif model_name.lower() == 'vgg16':
        if os.path.exists(path):
            model = load_model(path)
        else:
            model = keras.applications.vgg16.VGG16(input_shape=img_shape,
                                                                include_top=False,
                                                                # We don't need the classification layer.
                                                                weights=weights)
            model.save(filepath=path, include_optimizer=True)
            print(model_name+' saved')
        preprocessing = keras.applications.vgg16.preprocess_input
    elif model_name.lower() == 'vgg19':
        if os.path.exists(path):
            model = load_model(path)
        else:
            model = keras.applications.vgg19.VGG19(input_shape=img_shape,
                                                                include_top=False,
                                                                # We don't need the classification layer.
                                                                weights=weights)
            model.save(filepath=path, include_optimizer=True)
            print(model_name + ' saved')
        preprocessing = keras.applications.vgg19.preprocess_input
    elif model_name.lower() == 'resnet101':
        if os.path.exists(path):
            model = load_model(path)
        else:
            model = keras.applications.resnet.ResNet101(input_shape=img_shape,
                                                                include_top=False,
                                                                # We don't need the classification layer.
                                                                weights=weights)
            model.save(filepath=path, include_optimizer=True)
            print(model_name + ' saved')
        preprocessing = keras.applications.resnet.preprocess_input
    elif model_name.lower() == 'resnet152':
        if os.path.exists(path):
            model = load_model(path)
        else:
            model = keras.applications.resnet.ResNet152(input_shape=img_shape,
                                                                include_top=False,
                                                                # We don't need the classification layer.
                                                                weights=weights)
            model.save(filepath=path, include_optimizer=True)
            print(model_name + ' saved')
        preprocessing = keras.applications.resnet.preprocess_input
    elif model_name.lower() == 'nasnetlarge':
        if os.path.exists(path):
            model = load_model(path)
        else:
            model = keras.applications.nasnet.NASNetLarge(input_shape=img_shape,
                                                                include_top=False,
                                                                # We don't need the classification layer.
                                                                weights=weights)
            model.save(filepath=path, include_optimizer=True)
            print(model_name + ' saved')
        preprocessing = keras.applications.nasnet.preprocess_input
    elif model_name.lower() == 'densenet201':
        if os.path.exists(path):
            model = load_model(path)
        else:
            model = keras.applications.densenet.DenseNet201(input_shape=img_shape,
                                                                include_top=False,
                                                                # We don't need the classification layer.
                                                                weights=weights)
            model.save(filepath=path, include_optimizer=True)
            print(model_name + ' saved')
        preprocessing = keras.applications.densenet.preprocess_input

    elif model_name.lower() == 'vgg16-hybrid1365.h5':
        model = load_model(path[:-len('.h5')])
        preprocessing = None
    elif model_name.lower() == 'food11':
        model = load_model(path)
        preprocessing = None
    else:
        raise ValueError("Model "+model_name+" not implemented")
    print("Charged " + model_name)
    return model, preprocessing