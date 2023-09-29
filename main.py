import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data

import matplotlib.pyplot as plt

from layer import *
from activation import *
from optimizer import *
from loss import *

nnfs.init()

X, y = sine_data()

dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
activation3 = Activation_Linear()

loss_function = Loss_MeanSquaredError()

optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)
accuracy_precision = np.std(y) / 250

# Train in loop
for epoch in range(10001):  
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    data_loss = loss_function.calculate(activation3.output, y)

    regularization_loss = loss_function.regularization_loss(dense1) + \
                          loss_function.regularization_loss(dense2) + \
                          loss_function.regularization_loss(dense3)
    
    loss = data_loss + regularization_loss

    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}, ' +
        f'data_loss: {data_loss:.3f}, ' +
        f'reg_loss: {regularization_loss:.3f}), ' +
        f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

# Validate the model
X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X, activation3.output)
plt.show()
