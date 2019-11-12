"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch import Tensor
from torch.optim import SGD

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU

    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    super(MLP, self).__init__()

    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.neg_slope = neg_slope

    if n_hidden:
      hidden_layers = []
      for in_features, out_features in zip(n_hidden, n_hidden[1:]):
        hidden_layers += [nn.Linear(in_features, out_features), nn.LeakyReLU(neg_slope)]
      print(hidden_layers)
      self.model = nn.Sequential(
        nn.Linear(n_inputs, n_hidden[0]),
        nn.LeakyReLU(neg_slope),
        *hidden_layers,
        nn.Linear(n_hidden[-1], n_classes)
      )
    else:
      self.model = nn.Sequential(
        nn.Linear(n_inputs, n_classes)
      )
    self.model.apply(self.init_normal)
    print(self.model)

    ########################
    # END OF YOUR CODE    #
    #######################

  def init_normal(self, layer):
    if type(layer) == nn.Linear:
      nn.init.normal_(layer.weight, mean=0, std=0.0001)
      layer.bias.data.fill_(0)

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.model(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out