import numpy as np

from layer import Layer_Input
from activation import Activation_Softmax
from activation import Activation_Softmax_Loss_CategoricalCrossentropy
from loss import Loss_CategoricalCrossentropy

# Model class
class Model:
    def __init__(self):
        # list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # adds layer to model
    def add(self, layer):
        self.layers.append(layer)

    # sets loss and optimizer
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # finalize the model
    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):

            # first layer - prev object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # last layer - next object will be the loss
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # check which layers are trainable
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            # update loss object w/ trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    # train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        self.accuracy.init(y)

        for epoch in range(1, epochs + 1):
            # do forward pass
            output = self.forward(X, training=True)

            # calc loss
            data_loss, regularization_loss = \
                self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            # get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            # optimize (update params)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
                
        if validation_data is not None:
            X_val, y_val = validation_data

            # calc loss after forward pass
            output = self.forward(X_val, training=False)
            loss = self.loss.calculate(output, y_val)

            # get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print(f'validation, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}')


    def forward(self, X, training):
        # first layer uses input_layer's output
        self.input_layer.forward(X, training)

        # performs forward pass in a chain on all layers
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # 'layer' is last layer in self.layers
        return layer.output
    
    def backward(self, output, y):

        # check for softmax classifier
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            # last layer backward is not called (this is
            # the Softmax activation), so we set 
            # dinputs of this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs
            
             # call backward method for each layer 
            # in reversed order
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # call backward method on loss, this sets 
        # dinputs for last layer to access
        self.loss.backward(output, y)

        # call backward method for each layer 
        # in reversed order
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)