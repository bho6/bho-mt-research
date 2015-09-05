# This script uses Theano to implement a function approximator via a feed-
# forward neural network with backpropagation. The input is a list of two-
# dimensional coordinates (probably sampled from function_sampler.py). The
# output is a graph of the resulting approximated continuous function using =
# gnuplot.
#
# Usage: python function_approximator.py samples min max
#
# Brian Ho
# bho6@jhu.edu
# 9/1/15

import numpy
import sys
import theano
import theano.tensor as tensor

# Number of inputs.
num_input_nodes = 1

# Number of nodes in the hidden layer.
num_hidden_nodes = 10

# Rate to apply the gradient for training.
learning_rate = 0.1

# Threshold before stopping gradient descent.
delta = 0.0001


# Obtain a random value for a weight for the final layer based on the size of
# the previous layer.
def get_random_fweight(n):
  bound = 1 / numpy.sqrt(n)
  return numpy.random.uniform(-bound, bound)


# Obtain a random value for a weight for a hidden layer based on the size of
# previous layer and the size of the next layer.
def get_random_hweight(n_p, n_n):
  bound = numpy.sqrt(6) / numpy.sqrt((n_p + n_n))
  return numpy.random.uniform(-bound, bound)


# Common code for handling errors.
def handle_error(msg):
  print 'error: {}'.format(msg)
  sys.exit(1)


# Common code for converting input. Throws an error if incorrect input.
def convert_input(input, func, name):
  try:
    return func(input)
  except Exception:
    handle_error(
        'improper type for command line argument \'{}\'.'.format(name))


# Main entry point of the script.
def main():
  # Check the parameters and read in the samples file.
  if len(sys.argv) != 4:
    handle_error('invalid number of parameters.')
  
  sample_points = []
  try:
    with open(sys.argv[1]) as sample_file:
      for line in sample_file:
        index = line.find(' ')
        left = float(line[:index])
        right = float(line[index + 1:])
        sample_points.append((left, right))
  except Exception:
    handle_error('missing or invalid samples file.')
  domain_min = convert_input(sys.argv[2], float, 'min domain')
  domain_max = convert_input(sys.argv[3], float, 'max domain')

  # Use Theano to begin training.
  inputs = tensor.dvector()

  # Initialize the weights and biases from a random distribution.
  # Layer 1.
  wm_1 = numpy.empty([num_input_nodes, num_hidden_nodes])
  for i in xrange(0, num_input_nodes):
    for j in xrange(0, num_hidden_nodes):
      # Take into account the bias node.
      wm_1[i, j] = get_random_hweight(num_input_nodes + 1, 1)
  bm_1 = numpy.empty([1, num_hidden_nodes])
  for i in xrange(0, num_hidden_nodes):
    bm_1[0, i] = get_random_hweight(num_input_nodes + 1, 1)

  # Layer 2.
  wm_2 = numpy.empty([num_hidden_nodes, 1])
  for i in xrange(0, num_hidden_nodes):
    wm_2[i, 0] = get_random_fweight(num_hidden_nodes)
  bm_2 = numpy.empty([1, 1])
  bm_2[0, 0] = get_random_fweight(num_hidden_nodes)

  # Create symbolic variables.
  w_1 = theano.shared(value=wm_1)
  b_1 = theano.shared(value=bm_1)
  w_2 = theano.shared(value=wm_2)
  b_2 = theano.shared(value=bm_2)
  h = tensor.nnet.sigmoid(tensor.dot(inputs, w_1) + b_1)
  y_pred = tensor.nnet.sigmoid(tensor.dot(h, w_2) + b_2)

  # Define a cost function and compute gradients.
  y = tensor.dvector()
  l2_norm = (y_pred - y) ** 2
  cost = l2_norm.sum()
  gw_1, gb_1, gw_2, gb_2 = tensor.grad(cost, [w_1, b_1, w_2, b_2])

  # Define the training function.
  train = theano.function(
      inputs=[inputs, y],
      outputs=[y_pred, cost],
      updates=(
        (w_1, w_1 - learning_rate * gw_1), (b_1, b_1 - learning_rate * gb_1),
        (w_2, w_2 - learning_rate * gw_2), (b_2, b_2 - learning_rate * gb_2)
      )
  )

  # Train until we reach convergence defined by some delta.
  for _ in xrange(0, 1000):
    gt_delta = False
    for num, sample in enumerate(sample_points):
      input_vector = numpy.array([sample[0]])
      output_vector = numpy.array([sample[1]])
      sample_pred, sample_cost = train(input_vector, output_vector)

  # Predict over the entire range and generate a plot in gnuplot data format.
  predict = theano.function([inputs], y_pred)
  current = domain_min
  while current <= domain_max:
    input_vector = numpy.array([current])
    output = predict(input_vector)
    print '{} {}'.format(current, output[0][0])
    current += (domain_max - domain_min) / 100.0
    

if __name__ == "__main__":
  main()
