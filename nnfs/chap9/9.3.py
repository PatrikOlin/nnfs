#!/usr/bin/env python3
import numpy as np

# Passed-in gradient from the next layer
# for the purpose of this example we're going to use
# a vector of 1s
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# one bias for each neuron
# biases are the row vector with the shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list
dbiases = np.sum(dvalues, axis=0, keepdims=True)

print(dbiases)
