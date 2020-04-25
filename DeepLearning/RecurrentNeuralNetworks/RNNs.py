from data_reader import QuickDrawDataset
from plots import plot, evaluate_model, save_prediction_examples
from auxiliars import save_model, name_of_experiment, short_name_of_experiment, get_reliable_parameters,\
                        create_directories
from callbacks import PlotingCallback, ExtractValidationExamplesCallback

import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Conv1D, Dropout, BatchNormalization, Bidirectional, GRU, SimpleRNN
from keras.metrics import top_k_categorical_accuracy

def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y, 3)

import numpy as np
import os
import time

#Function corresponding to each Reccurrent Neuron Type
layer_correspondencies = {'RNN':SimpleRNN, 'GRU':GRU, 'LSTM': LSTM}

#Network parameters
#(Neurons and dropout_rate can be expressed as a list (or tuple) or as a unique value for all layers).
convolutional_parameters = {'use': True,
                            'layers':1,
                            'neurons':[64],
                            'dropout_rate':0.2,
                            'activation':'relu',
                            'kernel_size': 1}

recurrent_parameters = {'type':['RNN', 'GRU', 'LSTM'][2],
                         'layers':4,
                         'neurons':[64, 64, 128, 128]
                         'bidirectional':True,
                         'dropout_rate':0.2}

dense_parameters = {'layers':2,
                    'neurons':[256,512],
                    'activation':'relu',
                    'dropout_rate':[0.2, 0.5]}

charge_model = None #Path of the model to charge or None if training from scratch
preprocess = True

#Experiment Parameters
val_split = 0.16667
learning_rate = 0.0075

gpus = 4
batch_size = 448*gpus
reduce_epochs_by_factor = gpus*16 #Each epochs train only with samples//reduce_epochs_by_factor samples (aiming to generate more complete loss and accuracy curves)
epochs = 10*reduce_epochs_by_factor
callbacks_period = gpus #Each callbacks_period epochs, callbacks are called (in order to extract results)


optimizer = keras.optimizers.Adam(learning_rate)#Adagrad(learning_rate)

#Dataset Parameters
dataset_directory = os.path.join('.','NumpyQuickDrawDataset','Recognizeds')
instances_per_class = 60000 #How many instances will be used(They will be divided into Train-Test)
classes_to_charge = 345 #Amount of classes to use (if greater than total will be updated to total)
predictions_to_save = 25 #Prediction examples to save

#Natural Language description of the network
name_of_experiment = name_of_experiment(convolutional_parameters=convolutional_parameters, recurrent_parameters=recurrent_parameters,
                         dense_parameters=dense_parameters, num_classes=classes_to_charge,
                         training_instances=instances_per_class,learning_rate=learning_rate,
                                        preprocessing=preprocess, batch_size=batch_size)
#Short description of the network for use as a file name.
model_name =  short_name_of_experiment(convolutional_parameters=convolutional_parameters,
                                          recurrent_parameters=recurrent_parameters,
                                          dense_parameters=dense_parameters, num_classes=classes_to_charge,
                                          training_instances=instances_per_class, learning_rate=learning_rate,
                                       preprocessing=preprocess, batch_size=batch_size)+\
               '-'+str(np.random.randint(0,10))

print(name_of_experiment)

def generate_model(convolutional_parameters, recurrent_parameters, dense_parameters, input_shape=(None, 3), num_classes=345):
    """
    Generate a recurrent model based in the parameters dictionaries defined above
    :param convolutional_parameters: Dictionary of convolutional layers parameters (with Use at False if convs not desired).
    :param recurrent_parameters: Dictionary of Recurrent Layers parameters
    :param dense_parameters: Dictionary of Dense Layers parameters
    :param input_shape: Tuple with the input shape
    :param num_classes: Num of neurons at output softmax
    :return:
        Generated Keras Model
    """

    # Transform possible abbreviations on parameter definitions to its complete form
    convolutional_parameters = get_reliable_parameters(convolutional_parameters)
    recurrent_parameters = get_reliable_parameters(recurrent_parameters)
    dense_parameters = get_reliable_parameters(dense_parameters)


    model = Sequential()
    model.add(BatchNormalization(input_shape = input_shape, name='input_batch_norm'))
    #Add Convolutionals if required
    if convolutional_parameters['use']:
        for i in range(convolutional_parameters['layers']):
            model.add(Conv1D(filters=convolutional_parameters['neurons'][i],
                             kernel_size=convolutional_parameters['kernel_size'],
                             activation=convolutional_parameters['activation']))
            if convolutional_parameters['dropout_rate'][i] > 0:
                model.add(Dropout(rate=convolutional_parameters['dropout_rate'][i],
                          name='dropout_of_'+model.layers[-1].name))

    #Add Reccurent Layers
    for i in range(recurrent_parameters['layers']):
        recurrent_layer = layer_correspondencies[recurrent_parameters['type'].upper()]
        recurrent_layer = recurrent_layer(recurrent_parameters['neurons'][i],
                                          return_sequences=i<(recurrent_parameters['layers']-1))
        if recurrent_parameters['bidirectional']:
            recurrent_layer = Bidirectional(recurrent_layer)
        model.add(recurrent_layer)

        if recurrent_parameters['dropout_rate'][i] > 0:
            model.add(Dropout(rate=recurrent_parameters['dropout_rate'][i],
                              name='dropout_of_'+model.layers[-1].name))

    #Add Dense Units
    for i in range(dense_parameters['layers']):
        model.add(Dense(units=dense_parameters['neurons'][i], activation=dense_parameters['activation']))
        if dense_parameters['dropout_rate'][i] > 0:
            model.add(Dropout(rate=dense_parameters['dropout_rate'][i],
                              name='dropout_of_'+model.layers[-1].name))

    model.add(Dense(num_classes, activation=(tf.nn.softmax),
                    name="output_dense_layer"))
    return model

if __name__ == '__main__':

    #Charges the datasets for train, validation and test
    dataset_tr, dataset_val, dataset_test = [QuickDrawDataset(data_folder=dataset_directory, instances_per_class=instances_per_class,
                                               classes_to_charge=classes_to_charge, batch_size=batch_size,
                                               partition=partition, validation_split=val_split, preprocessing=preprocess)
    	for partition in ('training', 'validation', 'test')]
    #Get the name of classes charged and the Input Shape
    classes, input_shape = dataset_tr.classes, dataset_tr.input_shape

    #Charge the required model or generate a new one if not
    if charge_model is not None:
        model = load_model(filepath=charge_model)
        print("Model "+charge_model+" Charged: ")
        model.summary()
    else:
        model = generate_model(convolutional_parameters=convolutional_parameters,
                               recurrent_parameters=recurrent_parameters,
                               dense_parameters=dense_parameters,
                               input_shape=input_shape, num_classes=len(classes))

        print("Model Generated: ")
        model.summary()

    #If more than one GPU required put the model in multi_gpu mode
    model_gpu = multi_gpu_model(model, gpus = gpus) if gpus > 1 else model
    #Compile the model taking the accuracy and top-3 accuracy metrics
    if charge_model is None:
        model_gpu.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', top_3_accuracy])

    #Create and return the directories where results will be saved
    results_dir, model_dir, examples_dir = create_directories(model_name=model_name)
    #Create the checkpoint callback
    checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, "CheckPoint.hdf5"),
                                 monitor='val_acc', verbose=1, save_best_only=True, period=callbacks_period)

    #Writes the Natural Language model description
    with open(os.path.join(model_dir, 'model_description.txt'), 'w') as f:
        f.write(name_of_experiment)

    #Fits the model saving each callback_period iterations its partial results. Also evaluates the fitting time
    start_time = time.time()
    history = model_gpu.fit_generator(generator=dataset_tr, epochs=epochs, steps_per_epoch=len(dataset_tr)//reduce_epochs_by_factor,
                                      validation_steps=len(dataset_val)//reduce_epochs_by_factor, verbose=1, validation_data=dataset_val,
                                      use_multiprocessing=False, workers=25, max_queue_size=10*10,
                                      callbacks=[checkpoint,
                                                 PlotingCallback(path=results_dir, period=callbacks_period,
                                                                             validation_data=dataset_val),
                                                 ExtractValidationExamplesCallback(path=examples_dir,
                                                                                   period=callbacks_period,
                                                                                   validation_data=dataset_val,
                                                                                   predictions_to_save=predictions_to_save)])
    time_spent = time.time()-start_time
    m, s = divmod(time_spent, 60)
    h, m = divmod(m, 60)

    #Takes the final and complete validation results
    _, accuracy_validation, accuracy_top_3 = model_gpu.evaluate_generator(generator=dataset_val, steps=len(dataset_val), verbose=0)

    #Save the final model
    save_model(model=model, path=model_dir)

    #Plots and save the final accuracy curves
    [os.remove(os.path.join(results_dir, file)) for file in os.listdir(results_dir)]
    used_metrics = [metric for metric in model.history.params['metrics'] if 'val' not in metric]
    for label in used_metrics:
        plot(history=history, path=results_dir, label=label)

    #Save a set of the final prediction examples
    [os.remove(os.path.join(examples_dir, file)) for file in os.listdir(examples_dir)]
    save_prediction_examples(path = examples_dir, model=model, dataset=dataset_val,
                             predictions_to_save = predictions_to_save)

    #Save the final evaluation of the model and its confusion matrices
    for dataset, split in zip((dataset_tr, dataset_val, dataset_test), ('train', 'validation', 'test')):
        evaluate_model(model=model_gpu, path=results_dir, dataset=dataset, split=split, target_names=classes)

    #Change the folder name adding the validation accuracy at first and training time at the end
    model_name = 'Acc-' + str(np.round(accuracy_validation, 3)) + '-'+str(np.round(accuracy_top_3, 3))+'-'+model_name[:-2]+\
                      '_Time - '+'{: d}-{: 02d}-{: 02d}'.format(int(h), int(m), int(s))
    os.rename(os.path.dirname(results_dir), os.path.join(os.path.dirname(os.path.dirname(results_dir)), model_name))