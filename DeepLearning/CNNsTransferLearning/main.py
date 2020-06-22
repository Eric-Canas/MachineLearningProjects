from embedded_functions import plot_embedded_space, INPUT, OUTPUT
from plots import plot, evaluate_model
from auxiliars import save_model, short_name_of_experiment, get_reliable_parameters,\
                        create_directories, get_model, IMPLEMENTED_NETS
from callbacks import PlotingCallback
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from embedded_functions import get_embedded_from_dataset
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout
from keras.metrics import top_k_categorical_accuracy
import matplotlib
matplotlib.use('Agg')

def top_3_accuracy(x,y): return top_k_categorical_accuracy(x,y, 3)

import numpy as np
import os

train_dataset_directory = os.path.join('..','Datasets','food-101','images','train')
test_dataset_directory = os.path.join('..','Datasets','food-101','images','test')

target_size = (224, 224)
models = ['MobileNet', 'vgg16-hybrid1365.h5', 'NasNetLarge','ResNet101', 'ResNet152', 'DenseNet201', 'VGG16', 'VGG19']
model_name = 'DenseNet201'
base_model, preprocessing = get_model(model_name=model_name,img_shape=target_size+(3,)) #The feature Extractor

first_train_only_with_embedded = False
get_embedded_plots = False
freeze_convs = False

charge_model = None
flatten_layer, reduce_opp = {'Average Pooling': (keras.layers.GlobalAveragePooling2D(), np.mean), 'Flatten': (keras.layers.Flatten(), None)}['Average Pooling']
#Top Level Classifier
#Network parameters
dense_parameters = {'layers':0,
                    'neurons':[],
                    'activation':'relu',
                    'dropout_rate':[]}
classes = 101

#Experiment Parameters
val_split = 0.16667
learning_rate = 0.1

gpus = 1
batch_size = 64*gpus
#reduce_epochs_by_factor = gpus*16 #Each epochs train only with samples//reduce_epochs_by_factor samples (aiming to generate more complete loss and accuracy curves)
epochs_full = 5
epochs_fc = 25
callbacks_period = 4 #Each callbacks_period epochs, callbacks are called (in order to extract results)


optimizer = 'Adam'#keras.optimizers.Adam(learning_rate)#Adagrad(learning_rate)


#Short description of the network for use as a file name.
model_name =  short_name_of_experiment(base_model_name=model_name, dense_parameters=dense_parameters, learning_rate=learning_rate,
                                       preprocessing=preprocessing is not None, trained_embedded=freeze_convs)
print(model_name)


def fuse_models(base_model,flatten_layer, dense_part):
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

    x = base_model.output
    x = flatten_layer(x)
    x = dense_part(x)

    return Model(base_model.input, x)

def generate_dense_part(dense_parameters, shape, num_classes=101):
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
    dense_parameters = get_reliable_parameters(dense_parameters)


    model = Sequential()
    #Add Dense Units
    for i in range(dense_parameters['layers']):
        model.add(Dense(units=dense_parameters['neurons'][i], activation=dense_parameters['activation']))
        if dense_parameters['dropout_rate'][i] > 0:
            model.add(Dropout(rate=dense_parameters['dropout_rate'][i],
                              name='dropout_of_'+model.layers[-1].name))

    model.add(Dense(num_classes, activation=(tf.nn.softmax),
                    name="output_dense_layer"))
    if shape is not None:
      model.build(input_shape=(None,shape))
    return model

if __name__ == '__main__':

    complete_results_dir, fc_results_dir, model_dir, embedded_space_dir = create_directories(model_name=model_name)

    datagen_train_val = ImageDataGenerator(preprocessing_function=preprocessing,validation_split=0.0025,
                                           rotation_range=90,width_shift_range=0.2,height_shift_range=0.2,
                                           brightness_range=(0.5,1.5), zoom_range=0.3, horizontal_flip=True,
                                           vertical_flip=True)

    dataset_val, dataset_train = [datagen_train_val.flow_from_directory(directory=train_dataset_directory,
                                                                        target_size=target_size, color_mode='rgb',
                                                                        classes=None, class_mode='categorical',
                                                                        batch_size=batch_size, shuffle=split == 'training', seed=0,
                                                                        save_to_dir=None, follow_links=False,
                                                                        subset=split, interpolation='bilinear')
                                    for split in ('validation','training')]

    datagen_test = ImageDataGenerator(preprocessing_function=preprocessing)
    dataset_test = datagen_test.flow_from_directory(directory=test_dataset_directory,
                                                    target_size=target_size, color_mode='rgb',
                                                    classes=None, class_mode='categorical',
                                                    batch_size=batch_size, shuffle=False, seed=0,
                                                    save_to_dir=None, follow_links=False,
                                                    interpolation='bilinear')

    model = load_model('CheckPoint.h5',custom_objects={'top_3_accuracy':top_3_accuracy})

    # Save the final evaluation of the model and its confusion matrices
    for dataset, split in zip((dataset_test,), ('test',)):
        evaluate_model(model=model, path=complete_results_dir, dataset=dataset, split=split,
                       target_names=list(dataset_train.class_indices.keys()))

    checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, "FullyConnectedCheckPoint.h5"),
                    monitor='val_acc', verbose=1, save_best_only=True, period=callbacks_period)

    if get_embedded_plots:
        embedded_val, embedded_train, embedded_test = [get_embedded_from_dataset(model=base_model,
                                                                              dataset=dataset,
                                                                              standardize_output=True,
                                                                              get_hot_encoded=False,
                                                                              reduce_operation=reduce_opp)
                                                    for dataset in ((dataset_val, 'validation'), (dataset_train, 'train'),
                                                                    (dataset_test, 'test'))]

        plot_embedded_space(train_dataset=embedded_train,val_dataset=embedded_val, test_dataset=embedded_test,
                                embedded_space_dir=embedded_space_dir)

    if first_train_only_with_embedded:

        fc_val, fc_train, fc_test = [get_embedded_from_dataset(model=base_model,
                                                                              dataset=dataset,
                                                                              standardize_output=False,
                                                                              get_hot_encoded=True,
                                                                              reduce_operation=reduce_opp)
                                                    for dataset in ((dataset_val, 'validation'), (dataset_train, 'train'),
                                                                    (dataset_test, 'test'))]
        fully_connected_part = generate_dense_part(dense_parameters=dense_parameters,
                                                   shape=fc_train[INPUT].shape[-1],
                                                   num_classes=classes)
        # If more than one GPU required put the model in multi_gpu mode
        fully_connected_part_gpu = multi_gpu_model(fully_connected_part,
                                                   gpus=gpus) if gpus > 1 else fully_connected_part

        fully_connected_part_gpu.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
                                         metrics=['acc', top_3_accuracy])
        print("FC_PART SUMMARY")
        fully_connected_part.summary()
        history = fully_connected_part_gpu.fit(fc_train[INPUT], fc_train[OUTPUT], batch_size=batch_size,
                                     epochs=epochs_fc,
                                     validation_data=fc_val,
                                     callbacks=[PlotingCallback(path=fc_results_dir, period=callbacks_period),
                                                checkpoint])

        complete_model = fuse_models(base_model=base_model, flatten_layer=flatten_layer,
                                     dense_part=complete_results_dir)

        complete_model_gpu = multi_gpu_model(complete_model, gpus=gpus) if gpus > 1 else complete_model
        complete_model_gpu.compile(optimizer=optimizer, loss='categorical_crossentropy',
                                   metrics=['acc', top_3_accuracy])

    else:

        fully_connected_part = generate_dense_part(dense_parameters=dense_parameters,
                        shape=None,
                         num_classes=classes)
        complete_model = fuse_models(base_model=base_model, flatten_layer=flatten_layer, dense_part=fully_connected_part)
        # Freeze all Convolutional layers
        for layer in base_model.layers:
            layer.trainable = not freeze_convs
        print("FC_PART SUMMARY")
        complete_model.summary()
        # If more than one GPU required put the model in multi_gpu mode
        complete_model_gpu = multi_gpu_model(complete_model, gpus=gpus) if gpus > 1 else complete_model
        complete_model_gpu.compile(optimizer=optimizer, loss='categorical_crossentropy',
                                   metrics=['acc', top_3_accuracy])
        checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, "CompleteNetCheckPoint.h5"),
                    monitor='val_acc', verbose=1, save_best_only=True, period=callbacks_period)
        history = complete_model_gpu.fit_generator(generator=dataset_train, epochs=epochs_fc,
                                                   steps_per_epoch=len(dataset_train),
                                                   validation_steps=len(dataset_val), verbose=1,
                                                   validation_data=dataset_val,
                                                   use_multiprocessing=False, workers=25, max_queue_size=10 * 10,
                                                   callbacks=[PlotingCallback(path=fc_results_dir,
                                                                              period=callbacks_period),
                                                              checkpoint])
                                                              
    complete_model_gpu.compile(optimizer=optimizer, loss='categorical_crossentropy',
                                   metrics=['acc', top_3_accuracy])
    # Unfreeze all Convolutional layers
    for layer in base_model.layers:
        layer.trainable = True

    print("COMPLETE PART SUMMARY")
    complete_model.summary()
    history = complete_model_gpu.fit_generator(generator=dataset_train, epochs=epochs_full,
                            steps_per_epoch=len(dataset_train),
                            validation_steps=len(dataset_val), verbose=1,
                            validation_data=dataset_val,
                            use_multiprocessing=False, workers=25, max_queue_size=10 * 10,
                            callbacks=[PlotingCallback(path=complete_results_dir, period=callbacks_period),
                                       checkpoint])

    save_model(model=complete_model, path=model_dir)

    # Save the final evaluation of the model and its confusion matrices
    for dataset, split in zip((dataset_train, dataset_val, dataset_test), ('train', 'validation', 'test')):
        evaluate_model(model=complete_model_gpu, path=complete_results_dir, dataset=dataset, split=split,
                       target_names=list(dataset_train.class_indices.keys()))

    """
    #Create and return the directories where results will be saved
    #Create the checkpoint callback
    checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, "CheckPoint.hdf5"),
                                 monitor='val_acc', verbose=1, save_best_only=True, period=callbacks_period)


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
    """