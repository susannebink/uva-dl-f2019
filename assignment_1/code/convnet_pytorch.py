"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """
  # Useful for debugging
  class Print(nn.Module):
      def __init__(self):
          super(ConvNet.Print, self).__init__()

      def forward(self, x):
          #print(x.shape)
          return x

  # Flatten the output of the conv layers
  class Flatten(nn.Module):
      def __init__(self):
        super(ConvNet.Flatten, self).__init__()

      def forward(self, x):
        return x.view(x.size(0), -1)

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    super(ConvNet, self).__init__()

    kernel_size = 3

    self.model = nn.Sequential(
      nn.Conv2d(n_channels, 64, kernel_size=kernel_size, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1),
      nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1),
      nn.Conv2d(128, 256, kernel_size=kernel_size, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=kernel_size, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1),
      nn.Conv2d(256, 512, kernel_size=kernel_size, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1),
      nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1),
      ConvNet.Flatten(),
      nn.Linear(512*1*1, n_classes),
    )
    
    ########################
    # END OF YOUR CODE    #
    #######################
  
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
