import numpy as np

# Common loss class
class Loss:
    # regularization loss calculation
    def regularization_loss(self, layer):

        # 0 by default
        regularization_loss = 0

        # L1 regularization - weights 
        # calculated only when factor greater than zero
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        
        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        # L1 regularization - biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
        # L2 regularization - biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss
    
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

class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        # clip pred to avoid dividing by zero
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # calc sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # gradient calc
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # normalize gradient
        self.dinputs /= samples