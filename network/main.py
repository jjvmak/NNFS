import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from network.layers import Layer_Dense
from network.relu import Activation_ReLU
from network.softmax import Activation_Softmax
from network.loss_categorical_crossentropy import Loss_CategoricalCrossentropy

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

# 1 Layer and activation
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# 2 Layer and activation
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# Make forward pass for first layer and activation
dense1.forward(X)
activation1.forward(dense1.output)

# Make forward pass for first layer and activation
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)

print(loss)

# calculate accuracy
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
# True evaluates to 1; False to 0
# correct predictions / n
accuracy = np.mean(predictions == y)
print(accuracy)