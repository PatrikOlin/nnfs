#!/usr/bin/env python3
import numpy as np

# example layer output
z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

# ReLU activations derivative
# create array filled with zeroes in the shape of our example output from neuron (z)
# then se the values related to the inputs greater than 0 as 1
drelu = np.zeros_like(z)
drelu[z > 0] = 1

print(drelu)

# the chain rule
drelu *= dvalues

print(drelu)
