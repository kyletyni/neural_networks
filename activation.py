import numpy as np
from loss import Loss_CategoricalCrossentropy

# ReLU Activation
class Activation_ReLU:

    # ouputs values are zero for negative inputs 
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # backward pass
    def backward(self, dvalues):
        # copy dvalues since dinputs is being changed
        self.dinputs = dvalues.copy()

        # make negative values have zero gradient
        self.dinputs[self.inputs <= 0] = 0

    # calculates predictions for outputs
    def predictions(self, outputs):
        return outputs

# Softmax Activation
class Activation_Softmax:

    # forward pass
    def forward(self, inputs, training):
        self.inputs = inputs

        # get the unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # backward pass
    def backward(self, dvalues):
        # create unintialized array
        self.dinputs = np.empty_like(dvalues)

        # enumerate over outputs and gradients
        for index, (single_output, single_values) in enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1, 1)

            # calculate jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_values)

    # calculates predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Softmax Classifier - combined softmax activation and cross entropy loss
# for faster backward pass
class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    # backward pass
    def backward(self, dvalues, y_true):
        # num of samples
        samples = len(dvalues)

        # make discrete values if the samples are one-hot encoded
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy values so they can be safely changed
        self.dinputs = dvalues.copy()

        # calculate and normalize gradient
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # derivative calc of output from sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs