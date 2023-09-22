import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

nnfs.init()

# Dense Layer
class Layer_Dense:
    
    # layer initialization
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward pass 
    def forward(self, inputs):
        # calculate output values from inputs
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass
    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU Activation
class Activation_ReLU:

    # ouputs values are zero for negative inputs 
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # backward pass
    def backward(self, dvalues):
        # copy dvalues since dinputs is being changed
        self.dinputs = dvalues.copy()

        # make negative values have zero gradient
        self.dinputs[self.inputs < 0] = 0


# Softmax Activation
class Activation_Softmax:

    # forward pass
    def forward(self, inputs):
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


# Common loss class
class Loss:
    # calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output, y):
        # calculates samples losses and returns mean loss
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Cross-Entropy Loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # number of samples in a batch
        num_samples = len(y_pred)

        # clip data to prevent div by zero
        # clip data to not drag mean to any particular value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # calc probabilities for target values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(num_samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_probs = -np.log(correct_confidences)
        return negative_log_probs
    
    # backward pass
    def backward(self, dvalues, y_true):
        # get number of samples and labels in each sample
        samples = len(dvalues)
        labels = len(dvalues[0])

        # if the labels are sparse, they become one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues
        # normalize gradient
        self.dinputs /= samples

# Softmax Classifier - combined softmax activation and cross entropy loss
# for faster backward pass
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # forward pass
    def forward(self, inputs, y_true):
        # output layer's activation function
        self.activation.forward(inputs)

        # set the output
        self.output = self.activation.output

        # calculate and return loss
        return self.loss.calculate(self.output, y_true)
    
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
    

X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3)
active1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.forward(X)
active1.forward(dense1.output)
dense2.forward(active1.output)

loss = loss_activation.forward(dense2.output, y)

# print first few samples and loss value
print(loss_activation.output[:5])
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

# Print accuracy
print('acc:', accuracy)

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
active1.backward(dense2.dinputs)
dense1.backward(active1.dinputs)

# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)