import keras
import numpy as np
import os
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

test_samples_per_class = 10000

class QuickDrawDataset(keras.utils.Sequence):
    def __init__(self, data_folder = 'QuickDrawNumpySimplifiedDataset/Recognizeds', instances_per_class = 50000,
                 classes_to_charge=5, batch_size = 16, partition = 'training', validation_split=0.1, preprocessing=True,
                 random_seed = 0):
        #Preparing for data reading
        folders = os.listdir(data_folder)
        classes_to_charge = min(classes_to_charge, len(folders))

        #Preparing for save X and Y
        self.classes = np.array([folders[i][:-len('.npy')] for i in range(classes_to_charge)])
        self.X, self.Y = [], []
        train_instances_per_class = int((1 - validation_split)*instances_per_class)
        validation_instances_per_class = int((validation_split)*instances_per_class)
        #Reading the data
        for i in range(classes_to_charge):
            if partition == 'validation':
                x = np.load(os.path.join(data_folder, folders[i]), allow_pickle=True)[train_instances_per_class:train_instances_per_class+validation_instances_per_class]
            elif partition == 'training':
                x = np.load(os.path.join(data_folder, folders[i]), allow_pickle=True)[:train_instances_per_class]
            elif partition == 'test':
                x = np.load(os.path.join(data_folder, folders[i]), allow_pickle=True)[-test_samples_per_class:]
            y = np.ones(shape=len(x), dtype=np.int)*i
            self.X.append(x)
            self.Y.append(y)

        #Formatting the data
        self.X, self.Y = np.array(self.X).flatten(), np.array(self.Y).flatten()
        self.one_hot_encoded_Y = to_categorical(y=self.Y, num_classes=classes_to_charge)
        if preprocessing:
            for i in range(len(self.X)):
                self.X[i][:,:2] /= np.max(self.X[0][:,:2], axis=0)
            print(partition.title()+" re-escaled to 1-1")

        #Shuffling the data
        np.random.seed(random_seed)
        indexes = np.arange(len(self.X), dtype=np.int)
        np.random.shuffle(indexes)
        self.X, self.Y, self.one_hot_encoded_Y = self.X[indexes], self.Y[indexes], self.one_hot_encoded_Y[indexes]

        #Saving the rest of parameters
        self.batch_size = batch_size
        self.input_shape = (None,self.X[0].shape[-1])
        print(str(len(self.X))+" "+partition.title()+" Instances Charged")

    def __len__(self):
        return len(self.X)//self.batch_size

    def __getitem__(self, idx):
        """
        Gets a batch of x, y (input and output)
        :param idx: idx of the batch to return
        :return:
        (x, y) batch
        """
        real_idx = idx * self.batch_size
        x = pad_sequences(self.X[real_idx:real_idx+self.batch_size], padding='post')
        return x, self.one_hot_encoded_Y[real_idx:real_idx+self.batch_size]

    def on_epoch_end(self):
        """
        At the end of each epochs shuffle the instances
        """
        indexes = np.arange(len(self.X), dtype=np.int)
        np.random.shuffle(indexes)
        self.X, self.Y, self.one_hot_encoded_Y = self.X[indexes], self.Y[indexes], self.one_hot_encoded_Y[indexes]