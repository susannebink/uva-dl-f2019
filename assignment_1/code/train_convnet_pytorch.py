"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
from torch.nn import CrossEntropyLoss
import torch
from torch.optim import Adam
import cifar10_utils
import math
import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

  cnet = ConvNet(n_channels=3, n_classes=10)

  # TEST BATCH SIZE, DECREASE IF YA PC CANT HANDLE THIS
  test_batch = 10000

  if FLAGS.cuda:
    cnet.model.cuda()

  loss_fn = CrossEntropyLoss()
  optimiser = Adam(cnet.model.parameters(), lr=FLAGS.learning_rate)

  accuracies = []
  losses = []
  train_accs = []
  train_losses = []

  # train the model
  for i in range(FLAGS.max_steps):

    cnet.model.train()

    x, y = cifar10['train'].next_batch(FLAGS.batch_size)
    
    if FLAGS.cuda:
      x = torch.tensor(x).cuda()
      y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long).cuda()
    else:
      x = torch.tensor(x)
      y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)

    out = cnet.forward(x)
    loss = loss_fn(out, y)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    print("train {} of {} loss {}".format(i, FLAGS.max_steps, loss))

    # Calculate train and test loss and accuracy
    if not (i % FLAGS.eval_freq):

      cnet.model.eval()
      print("evaluating...")

      with torch.no_grad():

        x, y = cifar10['test'].next_batch(test_batch)
        if FLAGS.cuda:
          x = torch.tensor(x).cuda()
          y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long).cuda()
        else:
          x = torch.tensor(x)
          y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)

        out = cnet.forward(x)
        loss = loss_fn(out, y)
        acc = accuracy(out, y)

        accuracies.append(acc)
        losses.append(loss)

        print("iteration: {} accuracy:{} loss: {} <-- TEST".format(i, acc, loss))

        x, y = cifar10['train'].next_batch(test_batch)

        if FLAGS.cuda:
          x = torch.tensor(x).cuda()
          y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long).cuda()
        else:
          x = torch.tensor(x)
          y = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)

        out = cnet.forward(x)
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
  plt.savefig("conv-accuracy-{}-{}-{}.png".format(FLAGS.max_steps, FLAGS.learning_rate, FLAGS.batch_size))
  plt.clf()
  plt.plot(np.linspace(0, FLAGS.max_steps/ FLAGS.eval_freq, FLAGS.max_steps/ FLAGS.eval_freq), losses, label="test loss")
  plt.plot(np.linspace(0, FLAGS.max_steps/ FLAGS.eval_freq, FLAGS.max_steps/ FLAGS.eval_freq), train_losses, label="train loss")
  plt.xlabel("iteration")
  plt.ylabel("loss")
  plt.legend()
  plt.savefig("conv-loss-{}-{}-{}.png".format(FLAGS.max_steps, FLAGS.learning_rate, FLAGS.batch_size))
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
  parser.add_argument('-cuda', action="store_true",
                      help='enable cuda')
  FLAGS, unparsed = parser.parse_known_args()

  main()