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
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU Activation
class Activation_ReLU:

    # ouputs values are zero for negative inputs 
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax Activation
class Activation_Softmax:

    # forward pass
    def forward(self, inputs):
        # get the unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

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

X, y = vertical_data(100, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

dense1 = Layer_Dense(2, 3)
active1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
active2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()


# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(2000):

    dense1.weights += 0.5 * np.random.randn(2, 3)
    dense1.biases  += 0.5 * np.random.randn(1, 3)
    dense2.weights += 0.5 * np.random.randn(3, 3)
    dense2.biases  += 0.5 * np.random.randn(1, 3)

    dense1.forward(X)
    active1.forward(dense1.output)
    dense2.forward(active1.output)
    active2.forward(dense2.output)

    loss = loss_function.calculate(active2.output, y)

    predictions = np.argmax(active2.output, axis=1)
    accuracy = np.mean(predictions==y)  

    if loss < lowest_loss:
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases  = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases  = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()


# Print accuracy
print('acc:', accuracy)
print(loss)