import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from network.layers import Layer_Dense
from network.relu import Activation_ReLU


X, y = spiral_data(samples=100, classes=3)

# Layer and activation
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Make forward pass
dense1.forward(X)

# Use activation for the layer output
activation1.forward(dense1.output)

# Print five first
print(activation1.output[:5])

