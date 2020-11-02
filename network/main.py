import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from network.layers import Layer_Dense
from network.relu import Activation_ReLU
from network.softmax import Activation_Softmax

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

# 1 Layer and activation
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# 2 Layer and activation
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Make forward pass for first layer and activation
dense1.forward(X)
activation1.forward(dense1.output)

# Make forward pass for first layer and activation
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])


