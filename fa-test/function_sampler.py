# This script implements a function sampler over a range. The user can specify
# a separate Python file that implements the 'func' function (maybe written in
# Theano). This script will then sample n number of points and output them to
# standard output in the form 'input1 output1\ninput2 output2\n...' one pair per
# line.
#
# Usage: python function_sampler.py script x_min x_max num
#
# Brian Ho
# bho6@jhu.edu
# 9/1/15

import sys
import random

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
  # Check and assign the correct parameters.
  if len(sys.argv) != 5:
    handle_error('invalid number of parameters.')

  function_script = sys.argv[1]
  domain_min = convert_input(sys.argv[2], float, 'min domain')
  domain_max = convert_input(sys.argv[3], float, 'max domain')
  num_samples = convert_input(sys.argv[4], int, 'number of samples')

  namespace = {}
  try:
    execfile(function_script, namespace)
    sample_function = namespace['function']
  except Exception:
    handle_error(
        'the input script must implement a function called \'function\' ' +
        'which takes a float as its only argument and returns a float.')

  # Sample n points from this distribution.
  outputs = []
  for _ in xrange(num_samples):
    sample_point = (random.random() * (domain_max - domain_min)) + domain_min
    outputs.append('{} {}'.format(sample_point, sample_function(sample_point)))
  print '\n'.join(outputs)


if __name__ == "__main__":
  main()