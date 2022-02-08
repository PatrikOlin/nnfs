#!/usr/bin/env python3

class Layer_Dense:

    # Layer init
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neuron)
        self.biases = np.zeros((1, neurons))

    # forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    # backward pass
    def backward(self, dvalues):
        # since we need to modify the original variable,
        # lets make a copy of the values first
        self.dinputs = dvalues.copy()

        # zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

# ReLU activation
class Activation_ReLU:

    #forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# softmax activation
class Activation_Softmax:

    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Common loss class
class Loss:

    # calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # forward pass
    def forward(self, y_pred, y_true):

        # number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent dicision by 0
        # clip both sides to not drag mean toward any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        #number of labels in each sample
        # we'll use the first sample to count them
        labels = len(dvalues[0])

        # if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues

        # normalize gradient
        self.dinputs = self.dinputs / samples
