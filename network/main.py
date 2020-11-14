import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
from network.layers import Layer_Dense
from network.relu import Activation_ReLU
from network.softmax import Activation_Softmax
from network.loss_categorical_crossentropy import Loss_CategoricalCrossentropy

nnfs.init()

X, y = vertical_data(samples=100, classes=3)

# 1 Layer and activation
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# 2 Layer and activation
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

# Very naive weight update routine
for iteration in range(10000):
    # Update network
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    # calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    # True evaluates to 1; False to 0
    # correct predictions / n
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print('New set of weights found, iteration: ', iteration,
              'loss: ', loss, 'acc: ', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        # Revert back
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
