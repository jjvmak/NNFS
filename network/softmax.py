import numpy as np


class Activation_Softmax:
    def forward(self, inputs):

        # Unnormalised probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalized probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


