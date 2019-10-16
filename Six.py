import numpy as np
from numpy import exp, array, random, dot


# Readme
# Let's create an artificial neural network with 1 hidden layer (3 layers total with the
# input and the output), the weights are choosen randomly at the beginning, then adjusted
# thanks to backpropagation. The activation function used here is the Sigmoid one.

class Layers_of_Neurons():  #
    def __init__(self, nb_neurons, nb_inputs):
        # nb_neurons : number of neurons in the layer
        # nb_inputs : number of inputs that each neuron has
        self.synaptic_weights = 2 * random.random((nb_inputs, nb_neurons)) - 1


class ArticifialNeuralNetwork():
    def __init__(self, L2, L3):
        self.L2 = L2
        self.L3 = L3

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # We calculate the sum of the weights then we pass this number through the
    # Sigmoid function to normalise them between 0 and 1

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def training(self, inputs_set, output_set, nb_epochs):
        # We pass the training inputs through the ann, and we repeat this process
        # until the end of the number of epochs
        for epochs in range(nb_epochs):
            layer2_output, layer3_output = self.processing(inputs_set)

            # Calculus of the error of the L3
            L3_error = output_set - layer3_output
            L3_delta = L3_error * self.sigmoid_derivative(layer3_output)

            # Calculus of the error of the L2 (looking at the weights)
            L2_error = L3_delta.dot(self.L3.synaptic_weights.T)
            L2_delta = L2_error * self.sigmoid_derivative(layer2_output)

            # How much we need adjustments ?
            L2_adjustments = inputs_set.T.dot(L2_delta)
            L3_adjustments = layer2_output.T.dot(L3_delta)

            # Actual adjustments of the weights
            self.L2.synaptic_weights += L2_adjustments
            self.L3.synaptic_weights += L3_adjustments

    def processing(self, inputs):
        layer2_output = self.sigmoid(dot(inputs, self.L2.synaptic_weights))
        layer3_output = self.sigmoid(dot(layer2_output, self.L3.synaptic_weights))
        return layer2_output, layer3_output

    def weights(self):
        print ("  Layer 2 : 5 neurons with 3 inputs each: ")
        print self.L2.synaptic_weights
        print ("  Layer 3 (output): 1 neuron with 5 inputs: ")
        print self.L3.synaptic_weights

if __name__ == "__main__" :
        random.seed(1)

        L2 = Layers_of_Neurons(5, 3)
        L3 = Layers_of_Neurons(1, 5)

        # Combine the layers
        ann = ArticifialNeuralNetwork(L2, L3)

        print ("FIRST STEP: Putting random synpatic weights: ")
        ann.weights()

        # Here is our training sets of inputs and outputs
        training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
        training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

        #Now we train the neural network on the training set, and we do it 50 000 times(epochs)
        ann.training(training_set_inputs,training_set_outputs,50000)

        print ("SECOND STEP: Now we have new synaptic weights: ")
        ann.weights()

        print ("THIRD STEP: New situation [0,1,1] ?: ")
        new_state, output = ann.processing(array([0,1,1]))
        print output