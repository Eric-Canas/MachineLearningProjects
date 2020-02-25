import keras

import math

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Add, Dense, Input, Activation, Conv2D, MaxPooling2D, Flatten

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

val_split = 0.1
target_size = (256, 256)
epochs = 50
batch_size = 128
neurons_input_exp = (5, 8)
num_layers_exp = (2,4)# next -> (4,6). next -> (6,8). Complete exploration -> (2, 8)
round_value = 3

HEIGHT_ID, WIDTH_ID, CHANNELS_ID = 0, 1, 2

tr_dataset_directory = './food-101/images/train'
ts_dataset_directory = './food-101/images/test'

target_names = os.listdir(tr_dataset_directory)
optimizer = keras.optimizers.Adagrad(0.001)

def generate_model(input_shape, neurons_input, num_layers, max_neurons = 1024, skip_connections=True):

    block_len = math.ceil(num_layers / np.log2(input_shape[WIDTH_ID]))
    blocks = num_layers//block_len
    inp = Input(shape=input_shape)
    X = inp

    for i in range(blocks):
        for block in range(block_len):
            X = Conv2D(filters=min(neurons_input*(2**(i)), max_neurons), kernel_size=3, strides=(1,1), padding='same', activation='relu', input_shape=input_shape)(X)

        if skip_connections and i > 0:
            Y = Conv2D(filters=min(neurons_input*(2**(i)), max_neurons), kernel_size=1, strides=(1,1), padding='same', activation='relu', input_shape=input_shape)(Y)
            X = Add()([X,Y])

        Y = MaxPooling2D(pool_size=(2, 2))(X)
        X = Y

    X = Flatten()(X)
    X = Dense(units=min(neurons_input*(2**(blocks-1)), max_neurons)*2, activation='relu')(X)
    X = Dense(units=min(neurons_input*(2**(blocks-1)), max_neurons), activation='relu')(X)
    X = Dense(len(target_names), activation=(tf.nn.softmax))(X)

    return Model(inp, X)

def plot(history, path, label, xlabel='Epochs'):

    over_train, over_val = history.history[label], history.history['val_' + label]

    plt.plot(over_train)
    plt.plot(over_val)
    plt.title('Model '+ label.title())
    plt.ylabel(label.title())
    plt.xlabel(xlabel)
    plt.legend(['Training', 'Validation'], loc='upper left')

    if not os.path.exists(path=path):
        os.mkdir(path=path)

    name = label.title() + '. Tr-' + str(np.round(over_train[-1],round_value)) + '. Val-' + str(np.round(over_val[-1],round_value)) + '.pdf'

    plt.savefig(os.path.join(path, name))
    plt.close()

def evaluate_model(path, dataset, split, target_names):

    loss, accuracy = model.evaluate_generator(generator=dataset, steps=len(dataset), verbose=0)

    with open(os.path.join(path, split + '-' + str(np.round(accuracy, round_value)) + '-summary.txt'), 'w') as summary:

        summary.write('### SUMMARY ###\n\n')
        summary.write('Loss:' + str(loss) + '\n')
        summary.write('Accuracy:' + str(accuracy) + '\n\n')

        # Confusion matrix
        y_test = dataset.classes[dataset.index_array]
        y_pred = model.predict_generator(dataset, steps=len(dataset))

        summary.write('Analysis of results\n')
        summary.write(classification_report(y_test, np.argmax(y_pred, axis=1), target_names=target_names))
        summary.write(str(confusion_matrix(y_test, np.argmax(y_pred, axis=1))))

if __name__ == '__main__':

    datagen_tr_vl = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, samplewise_center=False, samplewise_std_normalization=False,
    	width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, brightness_range=(0.8, 1.2), validation_split=val_split)

    dataset_tr, dataset_vl = [datagen_tr_vl.flow_from_directory(directory=tr_dataset_directory, target_size=target_size, color_mode='rgb', classes=None, 
    	class_mode='categorical', batch_size=batch_size, shuffle=True, seed=0, save_to_dir=None, follow_links=False, subset=split, interpolation='bilinear') 
    	for split in ('training', 'validation')]

    datagen_preprocess = np.array([[dataset_tr[i][0].mean(axis=0), dataset_tr[i][0].std(axis=0)]
                                     for i in range(len(dataset_tr)-1)]).mean(axis=0)
    
    datagen_tr_vl.mean, datagen_tr_vl.std = datagen_preprocess[0], datagen_preprocess[1]

    datagen_ts = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, samplewise_center=False, samplewise_std_normalization=False)
    datagen_ts.mean, datagen_ts.std = datagen_tr_vl.mean, datagen_tr_vl.std

    dataset_ts = datagen_ts.flow_from_directory(directory=ts_dataset_directory, target_size=target_size, color_mode='rgb', classes=None, class_mode='categorical',
        batch_size=batch_size, shuffle=False, seed=0, save_to_dir=None, follow_links=False, interpolation='bilinear')

    input_shape = dataset_tr.image_shape

    for neurons_input_exp_it in range(neurons_input_exp[0], neurons_input_exp[1]):
        neurons_input = 2**neurons_input_exp_it

        for num_layers_exp_it in range(num_layers_exp[0], num_layers_exp[1]):
            num_layers = 2**num_layers_exp_it

            for skip_connections in (False,):
                model = generate_model(input_shape=input_shape, neurons_input=neurons_input, num_layers=num_layers, skip_connections=skip_connections)

                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                history = model.fit_generator(generator=dataset_tr, epochs=epochs, steps_per_epoch=len(dataset_tr), validation_steps=len(dataset_vl), verbose=1,
                	validation_data=dataset_vl, use_multiprocessing=True, workers=20, max_queue_size=10*10)

                _, accuracy_validation = model.evaluate_generator(generator=dataset_vl, steps=len(dataset_vl), verbose=0)

                matplotlib.use('Agg')
                model_name = 'Acc-' + str(np.round(accuracy_validation, round_value)) + \
                             '_neurons_input-' + str(neurons_input) + '_num_layers-' + str(num_layers) + \
                             '_SkipConnections(Add)-' + str(skip_connections)

                path = os.path.join('results', model_name)

                for label in ('acc', 'loss'):
                    plot(history=history, path=path, label=label)

                with open(os.path.join(path,'model.json'), 'w') as json_file:
                    json_file.write(model.to_json())

                weights_file = os.path.join(path, 'weights' + ".hdf5")
                model.save_weights(weights_file, overwrite=True)

                for dataset, split in zip((dataset_tr, dataset_vl, dataset_ts), ('train', 'validation', 'test')):
                    evaluate_model(path, dataset, split, target_names)