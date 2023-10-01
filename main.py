import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data

import matplotlib.pyplot as plt

from layer import *
from activation import *
from optimizer import *
from loss import *
from model import *
from accuracy import *

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# y = y.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)

model = Model()

model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
