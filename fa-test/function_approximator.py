# This script uses Theano to implement a function approximator via a feed-
# forward neural network with backpropagation. The input is a list of two-
# dimensional coordinates (probably sampled from function_sampler.py). The
# output is a graph of the resulting continuous function using gnuplot.
#
# Usage: python function_approximator.py samples
#
# Brian Ho
# bho6@jhu.edu
# 9/1/15

import sys

# Common code for handling errors.
def handle_error(msg):
  print 'error: {}'.format(msg)
  sys.exit(1)


# Main entry point of the script.
def main():
  # Check the parameters and read in the samples file.
  if len(sys.argv) != 2:
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

  print sample_points


if __name__ == "__main__":
  main()