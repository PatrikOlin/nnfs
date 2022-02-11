#!/usr/bin/env python3
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # remeber input values
        self.inputs = inputs
        #calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:

    def forward(self, inputs):
        # remember input values
        self.inputs = inputs

        #calculate output values from inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let’s make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

    def forward(self, inputs):
        # remember input values
        self.inputs = inputs

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):

        # create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1, 2)
            # calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)

            #calculate sample-wise gradient
            #and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# common loss class
class Loss:

    # calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # calculate sample losses
        sample_losses = self.forward(output, y)

        # calculate mean loss
        data_loss = np.mean(sample_losses)

        # return the loss
        return data_loss

# cross-entropy loss
class Loss_Categorical_Crossentropy(Loss):

    def forward(self, y_pred, y_true):

        #numer of samples in a batch
        samples = len(y_pred)

        # clip data to prevent divison by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 4 - 1e-7)

        # probabilities for targetr values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):

        # number of samples
        samples = len(dvalues)

        #number of labels in every sample
        # we'ull use the first sample to count them
        labels = len(dvalues[0])

        # if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues
        # normalize gradient
        self.dinputs = self.dinputs / samples

# softmax classifier - combined softmax activation
# and cross-entroyp loss for faster backward step
class Activation_Softmax_Loss_Categorical_Crossentropy:

    # create activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Crossentropy()

    def forward(self, inputs, y_true):
        # output layers activation function
        self.activation.forward(inputs)
        # set the output
        self.output = self.activation.output
        # calculate and return the loss value
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):

        # number of samples
        samples = len(dvalues)

        # if labels are one-hot encoded
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy so we can safely modify
        self.dinputs = dvalues.copy()

        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1

        # normalize gradient
        self.dinputs = self.dinputs / samples


# create dataset
X, y = spiral_data(samples=100, classes=3)

#create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

# create a second dense layer with 3 input features (as we take output
# of the previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# creat esoftmax classifiers combined loss and activation
loss_activation = Activation_Softmax_Loss_Categorical_Crossentropy()

# perform a forward pass of our training data through this layer
dense1.forward(X)

# perfrom a forward pass through activation function
# takes the output of the first dense layer
activation1.forward(dense1.output)

# perform a forward pass through second dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# perform a forwward pass through the activation/loss function
# takes the output of second dens layer here and returns loss
loss = loss_activation.forward(dense2.output, y)

# lets see the output of the first few samples:
print(loss_activation.output[:5])

# print loss value
print('loss: ', loss)

# calculate accuracy from output of activation2 and targets
# calculate calues along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

# print accuracy
print('acc: ', accuracy)

# backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
