import torch
import os
import numpy as np
from ASHRAEDataset import IDX_GROUPS
def save_model(model, path='./Models', file_name='default_model.pth'):
    path = os.path.join(path, file_name+'.pth')
    torch.save(model.state_dict(), path)

def load_model(model, path='./AcceptableModels', file_name='default_model.pth'):
    path = os.path.join(path, file_name)
    model.load_state_dict(torch.load(path))
    return model

def pearson_correlation(x,y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def pearsons_of_each_variable(data, idx_groups = IDX_GROUPS, device = 'cuda:0'):
    pearsons = []
    with torch.no_grad():
        for i, (x,y) in enumerate(data):
            x = x.to(device)
            y = y.to(device)
            pearsons_row = []
            for group in idx_groups:
                if type(group) is int:
                    pearson = pearson_correlation(x=x[..., group], y=y).item()
                else:
                    pearson = pearson_correlation(x=x[...,group].argmax(dim=-1).float(), y=y).item()
                pearsons_row.append(pearson)
            pearsons.append(pearsons_row)
            print('Pearson of Batch: '+str(pearsons_row))
    mean_pearson = np.mean(np.array(pearsons), axis=0)
    print(np.round(mean_pearson,6))
    return mean_pearson

def RMSLELoss():
    def RMSLE(output, target):
        return torch.sqrt(torch.sum((torch.log(output+1)-torch.log(target+1))**2)/target.size()[0])
    return RMSLE