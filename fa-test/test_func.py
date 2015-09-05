# This implements a test function to be used as input in function_sampler. More
# formally, this uses the sine function.
#
# Brian Ho
# bho6@jhu.edu
# 9/1/15

import math

# All valid inputs to function_sampler.py need to implement a function named
# 'function'. The function takes in a singular float as an argument and returns
# another float.
def function(num):
  return math.sin(num)
