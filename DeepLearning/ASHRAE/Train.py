from ASHRAEDataset import ASHRAEDataset
import torch
import Models as models
from torch.utils import data
import warnings
import math
from torch import nn
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter
from NetworkUtils import save_model, load_model, pearson_correlation, pearsons_of_each_variable, RMSLELoss
from torch.utils.data import DataLoader
import time
import os

def train_step(model, data, epoch, criterion, optimizer, device = 'cuda:0', verbose = False, writer=None, verbose_each=20000):
    model.train()
    losses = []
    pearsons = []
    datalen = len(data)
    batch_time = time.time()
    for i, (x,y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        output = model(x)[...,0]
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pearson = pearson_correlation(x=output, y=y).item()
        pearsons.append(pearson)
        if verbose and i % verbose_each == 0:
            this_batch_time = time.time()-batch_time
            percent_done = i/datalen
            remaining_time = (this_batch_time*(datalen/verbose_each))*(1-percent_done)
            sample_id = np.random.randint(0, 2)
            sample_id_2 = np.random.randint(4, 8)
            sample_id_3 = np.random.randint(2, 4)
            print('Train -> Batch {0}/{1} ({2}%) of epoch {3}. Loss: {4}, Pearson: {5}, STD: {6}, Samples: {7}/{8} - {9}/{10} - {11}/{12} --> Remaining Time: {13}:{14}'.
                  format(i,datalen, int((percent_done)*100), epoch, np.round(loss.item(),3), np.round(pearson,3),
                         np.round(torch.std(output).item(),2), np.round(output[sample_id].item(),2),
                         np.round(y[sample_id].item(),2),np.round(output[sample_id_2].item(),2),
                         np.round(y[sample_id_2].item(),2),np.round(output[sample_id_3].item(),2),
                         np.round(y[sample_id_3].item(),2),
                         int(remaining_time/60),int(remaining_time)%60))
            batch_time = time.time()
    mean_losses = np.mean(np.array(losses))
    mean_pearsons = np.mean(np.array(pearsons))
    if writer is not None:
        writer.add_scalar('Train-Loss', mean_losses, epoch)
        writer.add_scalar('Train-Pearson', mean_pearsons, epoch)
    print('TRAIN -> EPOCH {0}, MEAN LOSS: {1}, MEAN PEARSON: {2}'.format(epoch, np.round(mean_losses,4), np.round(mean_pearsons,3)))

def validation_step(model, data, epoch, criterion, device = 'cuda:0', writer=None):
    model.eval()
    losses = []
    pearsons = []
    with torch.no_grad():
        for i, (x,y) in enumerate(data):
            x = x.to(device)
            y = y.to(device)
            output = model(x)[...,0]
            loss = criterion(output, y)
            losses.append(loss.item())
            pearson = pearson_correlation(x=y, y=output).item()
            pearsons.append(pearson)
    mean_loss = np.mean(losses)
    mean_pearson = np.mean(np.array(pearsons))
    if writer is not None:
        writer.add_scalar('Validation-Loss', mean_loss, epoch)
        writer.add_scalar('Validation-Pearson', mean_pearson, epoch)
    print('VALIDATION -> EPOCH {0}, MEAN LOSS {1}, STD LOSS {2}, MEAN PEARSON {3}'.format(epoch, np.round(mean_loss,4),
                                                                                np.round(np.std(np.array(losses)),3),
                                                                                np.round(mean_pearson,3)))

def get_predictions(model, data, batch_size, device = 'cuda:0', verbose=True, verbose_each=200):
    model.eval()
    datalen = len(data)
    batch_time = time.time()
    output_tensor = torch.zeros((len(data.dataset)),device=device)
    #output_tensor[:,0] = torch.arange(0,len(data.dataset),device=device)
    with torch.no_grad():
        for i, (x,_) in enumerate(data):
            x = x.to(device)
            output = model(x)[...,0]
            output_tensor[i * batch_size:(i + 1) * batch_size] = output
            if verbose and i % verbose_each == 0:
                this_batch_time = time.time() - batch_time
                percent_done = i / datalen
                remaining_time = (this_batch_time * (datalen / verbose_each)) * (1 - percent_done)
                print('Producing Output -> Batch {0}/{1} ({2}%) --> Remaining Time: {3}:{4}'.
                      format(i, datalen, int((percent_done) * 100),
                             int(remaining_time / 60), int(remaining_time) % 60))
                batch_time = time.time()
    return output_tensor.detach().cpu().numpy()

def train(lr=0.01, gpu = 0, epochs = 500, file_name='DefaultFileName',
                 charge=None, save = True, batch_size = 64, epochs_for_saving=3):

    #Stablishing the device
    device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warnings.warn(message="Executing on CPU!", category=ResourceWarning)

    #Generating the model
    #Change this line for changing the model to create
    model = models.ThreeLayerSigmoidRegressor()
    if charge is not None:
        model = load_model(model=model,file_name=charge)
    model = model.to(device)

    #Training Parameters
    criterion = RMSLELoss()#nn.MSELoss()#nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    #Generating the Dataset
    dataset = ASHRAEDataset(erase_nans=False)
    train_len, validation_len = int(math.ceil(0.95 * len(dataset))), int(math.ceil(0.04 * len(dataset)))
    train, validation, test = data.random_split(dataset, (train_len,
                                                          validation_len,
                                                          len(dataset)-train_len-validation_len))
    #Pass to Dataloader for reading batches
    train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    validation = DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    #Uncomment for seeing pearson correlations
    #pearsons_of_each_variable(data=train)

    #Writer for plotting graphic in tensorboard
    writer = SummaryWriter(comment=file_name)
    print('Starting the training...')
    print("Batch Size: " + str(batch_size))
    print("Running in: " + device)



    for i in range(epochs):
        train_step(model=model, data=train, criterion=criterion, optimizer=optimizer, epoch=i, device=device,
                   writer=writer, verbose=True)
        validation_step(model=model, data=validation, criterion=criterion, epoch=i, device=device, writer=writer)
        if save and i%epochs_for_saving==0:
            writer.flush()
            model = model.cpu()
            save_model(model=model,file_name=file_name)
            model = model.to(device)
    writer.close()

def produce_test_output(model_to_charge, gpu = 0, batch_size=2000):
    # Stablishing the device
    device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warnings.warn(message="Executing on CPU!", category=ResourceWarning)

    # Charging the model
    # Change this line for changing the model to create
    model = models.ThreeLayerSigmoidRegressor()
    model = load_model(model=model, file_name=model_to_charge)
    model = model.to(device)

    # Generating the Dataset
    dataset = ASHRAEDataset(charge_train=True, charge_test=True)
    dataset.charge = 'Test'
    # Pass to Dataloader for reading batches
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Writer for plotting graphic in tensorboard
    print('Starting the prediction...')
    print("Batch Size: " + str(batch_size))
    print("Running in: " + device)

    predictions_without_id = get_predictions(model=model,data=dataset, batch_size=batch_size, device=device)
    predictions = np.zeros(shape=(len(predictions_without_id),2), dtype=np.float64)
    predictions[...,0] = np.arange(0,len(predictions), dtype=np.float64)
    predictions[...,1] = predictions_without_id

    np.savetxt('./Submisions/submision_'+model_to_charge[:-len('.pth')]+'.csv',predictions,delimiter=',',
               fmt=['%u','%1.4f'],header='row_id,meter_reading')

