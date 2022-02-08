#!/usr/bin/env python3
import numpy as np

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1, 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# sum weights of the given input
# and multiply by the passed-in gradient for this neuron
dweights = np.dot(inputs.T, dvalues)

print(dweights)
