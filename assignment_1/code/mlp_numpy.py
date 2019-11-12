"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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

    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.neg_slope = neg_slope

    self.network = []

    if self.n_hidden:
      
      self.network.append(LinearModule(n_inputs, n_hidden[0]))
      self.network.append(LeakyReLUModule(neg_slope))

      for i in range(len(n_hidden[1:])):
        self.network.append(LinearModule(n_hidden[i], n_hidden[i + 1]))
        self.network.append(LeakyReLUModule(neg_slope))

      self.network.append(LinearModule(n_hidden[-1], n_classes))
      self.network.append(SoftMaxModule())

    else:
      self.network.append(LinearModule(n_inputs, n_classes))
      self.network.append(SoftMaxModule())


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
    for module in self.network:
      x = module.forward(x)
    out = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    for module in self.network[::-1]:
      dout = module.backward(dout)
    ########################
    # END OF YOUR CODE    #
    #######################

    return
