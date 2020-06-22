# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
import random
import torch.autograd as autograd
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("Using GPU Environment")
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    #if self.training:
    #  return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    #else:
    return F.linear(input, self.weight_mu, self.bias_mu)


class Raimbow(nn.Module):
  def __init__(self, input_size, num_actions, architecture = 'data-efficient', prepare_decoder=None):
    super(Raimbow, self).__init__()
    self.atoms = 51
    self.action_space = num_actions

    if architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(input_size[0], 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(input_size[0], 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    #self.fc_h_v = NoisyLinear(self.conv_output_size, 256, std_init=0.1)
    #self.fc_h_a = NoisyLinear(self.conv_output_size, 256, std_init=0.1)
    #self.fc_z_v = NoisyLinear(256, self.atoms, std_init=0.1)
    #self.fc_z_a = NoisyLinear(256, num_actions * self.atoms, std_init=0.1)
    #self.pre_output = nn.Linear(num_actions*self.atoms,64)
    self.pre_output = nn.Linear(self.conv_output_size, 256)
    self.output = nn.Linear(256, num_actions)



  def forward(self, x, log=False):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    #v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    #a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    #v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    #q = v + a - a.mean(1, keepdim=True)  # Combine streams
    #if log:  # Use log softmax for numerical stability
    #  q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    #else:
    #  q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    x = F.relu(self.pre_output(x))
    return self.output(x)

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

  def act(self, state, epsilon):
    if random.random() > epsilon:
      state = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False)
      q_value = self.forward(state)
      action = q_value.max(1)[1].item()
    else:
      action = random.randrange(self.action_space)
    return action
