"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP, CrossEntropyLoss, Tensor, SGD, Adam
import torch
import cifar10_utils
import sys
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  correct = (np.argmax(predictions.cpu().numpy(), axis=1) == targets.cpu().numpy()).sum() / targets.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################
  return correct

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope
  
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

  mlp = MLP(3072, dnn_hidden_units, 10, neg_slope)
  if FLAGS.cuda:
    mlp.model.cuda()
  loss_fn = CrossEntropyLoss()
  optimiser = Adam(mlp.model.parameters(), lr=FLAGS.learning_rate)

  accuracies = []
  losses = []
  train_accs = []
  train_losses = []
  for i in range(FLAGS.max_steps):

    mlp.model.train()

    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    if FLAGS.cuda:
      x = torch.tensor(np.reshape(x, (FLAGS.batch_size, 3072))).cuda()
      y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long).cuda()
    else:
      x = torch.tensor(np.reshape(x, (FLAGS.batch_size, 3072)))
      y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)
    
    out = mlp.forward(x)
    loss = loss_fn(out, y)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if not (i % FLAGS.eval_freq):

      mlp.model.eval()

      with torch.no_grad():

        x, y = cifar10['test'].next_batch(10000)
        if FLAGS.cuda:
          x = torch.tensor(np.reshape(x, (10000, 3072))).cuda()
          y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long).cuda()
        else:
          x = torch.tensor(np.reshape(x, (10000, 3072)))
          y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)
      
        out = mlp.forward(x)
        loss = loss_fn(out, y)

        acc = accuracy(out, y)
        accuracies.append(acc)
        losses.append(loss)

        print("iteration: {} accuracy:{} loss: {} <--- TRAIN".format(i, acc, loss))

        x, y = cifar10['train'].next_batch(10000)
        if FLAGS.cuda:
          x = torch.tensor(np.reshape(x, (10000, 3072))).cuda()
          y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long).cuda()
        else:
          x = torch.tensor(np.reshape(x, (10000, 3072)))
          y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)
        out = mlp.forward(x)
        loss = loss_fn(out, y)

        acc = accuracy(out, y)
        train_accs.append(acc)
        train_losses.append(loss)

        print("iteration: {} accuracy:{} loss: {}".format(i, acc, loss))


  plt.plot(np.linspace(0, FLAGS.max_steps / FLAGS.eval_freq, FLAGS.max_steps/ FLAGS.eval_freq), accuracies, label="test accuracy")
  plt.plot(np.linspace(0, FLAGS.max_steps / FLAGS.eval_freq, FLAGS.max_steps/ FLAGS.eval_freq), train_accs, label="train accuracy")
  plt.xlabel("iteration")
  plt.ylabel("accuracy")
  plt.legend()
  plt.savefig("pytorch-accuracy-{}-{}-{}.png".format(FLAGS.max_steps, FLAGS.learning_rate, FLAGS.batch_size))
  plt.clf()
  plt.plot(np.linspace(0, FLAGS.max_steps/ FLAGS.eval_freq, FLAGS.max_steps/ FLAGS.eval_freq), losses, label="test loss")
  plt.plot(np.linspace(0, FLAGS.max_steps/ FLAGS.eval_freq, FLAGS.max_steps/ FLAGS.eval_freq), train_losses, label="train loss")
  plt.xlabel("iteration")
  plt.ylabel("loss")
  plt.legend()
  plt.savefig("pytorch-loss-{}-{}-{}.png".format(FLAGS.max_steps, FLAGS.learning_rate, FLAGS.batch_size))
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  parser.add_argument('-cuda', action="store_true",
                      help='enable cuda')
  FLAGS, unparsed = parser.parse_known_args()

  main()