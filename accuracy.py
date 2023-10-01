import numpy as np

# Common Accuracy Class
class Accuracy:

    # calculates and returns an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):
        # get comparisons and accuracy
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        return accuracy
        
class Accuracy_Regression(Accuracy):

    def __init__(self):
        # creates precision property 
        self.precision = None

    # calculates precision value based on 
    # ground truth values
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # compare predictions to ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    

class Accuracy_Categorical(Accuracy):
    # no initialization needed
    def init(self, y):
        pass

    # compare predictions to ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
