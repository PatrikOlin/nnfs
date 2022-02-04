#!/usr/bin/env python3

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate outputs values from inputs, weights and biass
        self.output = np.dot(inputs, self.weights) + self.biases

# Rectified linear activation function
class Activation_ReLU:

    #forward pass
    def forward(self, inputs):
        # calculate output values from inputs
        self.output = np.maximum(0, inputs)

# softmax activation
class Activation_Softmax:

    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

# Create second dense layer with 3 input features (as we take
# output of previous layer) and 3 output values
dense2 = Layer_Dense(3, 3)

# create softmax activation (to be used with dense layer)
activation2 = Activation_Softmax()

# Perforom a forward pass of our training data through this layer
dense1.forward(X)

# make a forward pass through actuvation function, which takes
# the output of the first dense layer
activation1.forward(dense1.output)

# make a forward pass through the second dense layer
# it takes outputs of activation function of first layer as imports
dense2.forward(activation1.output)

# make a forward pass through activation function
# it takes the output of the second dense layer
activation2.forward(dense2.output)

# output of first few samples
print(activation2.output[:5])
