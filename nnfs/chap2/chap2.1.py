#!/usr/bin/env python3
import numpy as np

# thse are both vectors (one dimensional lists)
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]

bias = 2.0

# dot product (np.dot) returns the sum of products of consecutive vector elements (a[0]*b[0] + a[1]*b[1] etc)
outputs = np.dot(weights, inputs) + bias

print(outputs)
