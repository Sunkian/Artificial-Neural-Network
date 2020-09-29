import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


print('SIG', sigmoid(1.57))


class FFNeuralNetwork():

    def __init__(self):
        np.random.seed(1)

    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    # def sigmoid_derivative(self, x):
    #     return x * (1 - x)

    X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    print("Shape of output:", X.shape)
    # X = X.T
    # print("Shape of output with T:", X.shape)
    print(X)

    Y = np.array([[0, 1, 1, 0]])
    Y = Y.T
    print("Shape of output:", Y.shape)
    print(Y)

    weights = 2 * np.random.rand(3, 1) - 1
    print('Poids1:', weights)
    bias = np.random.rand(1)
    lr = 0.05

    for epoch in range(2000):
        input_hidden = np.dot(X, weights) + bias
        output_hidden = sigmoid(input_hidden)

        error = output_hidden - Y

        output_delta = error * sigmoid_derivative(output_hidden)

        inputs = X.T
        weights -= lr * np.dot(inputs, output_delta)

        for num in output_delta:
            bias -= lr * num

        # print("Error:", error.sum())

        test = np.array([0, 1, 1])
        result = sigmoid(np.dot(test, weights) + bias)
        print("RESULTAT:", result)
