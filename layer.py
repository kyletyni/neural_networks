import numpy as np

# Dense Layer
class Layer_Dense:
    
    # layer initialization
    def __init__(self, n_inputs, n_neurons, 
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # forward pass 
    def forward(self, inputs, training):
        # calculate output values from inputs
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward pass
    def backward(self, dvalues):
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)

# Dropout Layer
class Layer_Dropout:

    # layer initialization
    def __init__(self, rate):
        # self.rate is the % of neurons that are not dropped
        self.rate = 1 - rate

    # forward pass
    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        # create scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        # apply the mask to generate outputs
        self.output = inputs * self.binary_mask

    # backward pass
    def backward(self, dvalues):
        # gradient on values
        self.dinputs = dvalues * self.binary_mask


# Input layer class
class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

    