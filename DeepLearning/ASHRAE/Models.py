import torch.nn as nn
import torch.nn.functional as F
from ASHRAEDataset import INPUT_LEN
import torch

class OneLayerRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN, hidden_size=INPUT_LEN*2, output_size=1):
        super(OneLayerRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)  # hidden layer
        self.output = nn.Linear(hidden_size, output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.output(x))  # linear output
        return x

class TwoLayerLinearRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN, INPUT_LEN//2), output_size=1):
        super(TwoLayerLinearRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.output = nn.Linear(hidden_size[1], output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.output(x))  # linear output
        return x

class ThreeLayerLinearRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2), output_size=1):
        super(ThreeLayerLinearRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.output = nn.Linear(hidden_size[2], output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.output(x))  # linear output
        return x

class FourLayerLinearRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2, INPUT_LEN//4), output_size=1):
        super(FourLayerLinearRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.hidden4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.output = nn.Linear(hidden_size[3], output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.output(x))  # linear output
        return x

class FiveLayerLinearRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*4, INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2, INPUT_LEN//4), output_size=1):
        super(FiveLayerLinearRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.hidden4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.hidden5 = nn.Linear(hidden_size[3], hidden_size[4])
        self.output = nn.Linear(hidden_size[4], output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.output(x))  # linear output
        return x

class SixLayerLinearRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*8, INPUT_LEN*4, INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2, INPUT_LEN//4), output_size=1):
        super(SixLayerLinearRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.hidden4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.hidden5 = nn.Linear(hidden_size[3], hidden_size[4])
        self.output = nn.Linear(hidden_size[4], output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.output(x))  # linear output
        return x


class FiveLayerSigmoidRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*4, INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2, INPUT_LEN//4), output_size=1):
        super(FiveLayerSigmoidRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.hidden4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.hidden5 = nn.Linear(hidden_size[3], hidden_size[4])
        self.output = nn.Linear(hidden_size[4], output_size)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = torch.sigmoid(self.hidden3(x))
        x = torch.sigmoid(self.hidden4(x))
        x = torch.sigmoid(self.hidden5(x))
        x = F.relu(self.output(x))  # linear output
        return x

class FourLayerSigmoidRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*4, INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2), output_size=1):
        super(FourLayerSigmoidRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.hidden4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.output = nn.Linear(hidden_size[3], output_size) # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = torch.sigmoid(self.hidden3(x))
        x = torch.sigmoid(self.hidden4(x))
        x = F.relu(self.output(x))  # linear output
        return x

class ThreeLayerSigmoidRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2), output_size=1):
        super(ThreeLayerSigmoidRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.output = nn.Linear(hidden_size[2], output_size)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = torch.sigmoid(self.hidden3(x))
        x = F.relu(self.output(x))  # linear output
        return x

class TwoLayerSigmoidRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN, INPUT_LEN//2), output_size=1):
        super(TwoLayerSigmoidRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.output = nn.Linear(hidden_size[1], output_size)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = F.relu(self.output(x))  # linear output
        return x

class OneLayerSigmoidRegressor(nn.Module):
    def __init__(self, input_size=INPUT_LEN, hidden_size=INPUT_LEN*2, output_size=1):
        super(OneLayerSigmoidRegressor, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)  # hidden layer
        self.output = nn.Linear(hidden_size, output_size)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.output(x))  # linear output
        return x


class ThreeLayerMixRegressorSRR(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2), output_size=1):
        super(ThreeLayerMixRegressorSRR, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.output = nn.Linear(hidden_size[2], output_size)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.output(x))  # linear output
        return x

class ThreeLayerMixRegressorRSR(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2), output_size=1):
        super(ThreeLayerMixRegressorRSR, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.output = nn.Linear(hidden_size[2], output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.output(x))  # linear output
        return x

class ThreeLayerMixRegressorSSR(nn.Module):
    def __init__(self, input_size=INPUT_LEN,
                 hidden_size=(INPUT_LEN*2, INPUT_LEN, INPUT_LEN//2), output_size=1):
        super(ThreeLayerMixRegressorSSR, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size[0])  # hidden layer
        self.hidden2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.output = nn.Linear(hidden_size[2], output_size)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.output(x))  # linear output
        return x