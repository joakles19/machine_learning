import numpy as np

class Neuron:
    def __init__(self, num_inputs, layer):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.layer = layer

    def sigmoid_activation(self, z):
        result = 1 / (1 + np.exp(-z))

        return result
    
    def sigmoid_derivative(self):
        result = self.activation * (1 - self.activation)

        return result
    
    def forward_pass(self, input_vector):
        self.inputs = input_vector
        self.z = np.dot(self.weights, input_vector) + self.bias
        self.activation = self.sigmoid_activation(self.z)

        return self.activation
    
class Layer:
    def __init__(self, num_neurons, inputs_per_neuron, layer):
        self.neurons = [Neuron(inputs_per_neuron, layer) for _ in range(num_neurons)]

    def forward_pass(self, input_vector):
        output_vector = []

        for neuron in self.neurons:
            output_vector.append(neuron.forward_pass(input_vector))

        return np.array(output_vector)
    
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, num_neurons, num_inputs):
        layer_index = len(self.layers)
        layer = Layer(num_neurons, num_inputs, layer_index)

        self.layers.append(layer)

    def forward_pass(self, input_vector):
        for layer in self.layers:
            input_vector = layer.forward_pass(input_vector)

        return input_vector
    
    def loss_function(self, true_values, predicted_values):
        loss = np.mean((true_values - predicted_values) ** 2)

        return loss
    
    def train(self, x, y, L=0.1):
        output = self.forward_pass(x)

        gradients = [None] * len(self.layers)

        last_layer = self.layers[-1]
        d_output = np.zeros(len(last_layer.neurons))

        for i, neuron in enumerate(last_layer.neurons):
            d_output[i] = (neuron.activation - y[i]) * neuron.sigmoid_derivative()
            neuron.weights -= L * d_output[i] * neuron.inputs
            neuron.bias -= L * d_output[i]

        gradients[-1] = d_output

        for l in reversed(range(len(self.layers) - 1)):
                layer = self.layers[l]
                next_layer = self.layers[l + 1]
                d_hidden = np.zeros(len(layer.neurons))

                for i, neuron in enumerate(layer.neurons):
                    errors = [next_neuron.weights[i] * gradients[l + 1][k] for k, next_neuron in enumerate(next_layer.neurons)]
                    error_sum = sum(errors)
                    d_hidden[i] = neuron.sigmoid_derivative() * error_sum
                    neuron.weights -= L * d_hidden[i] * neuron.inputs
                    neuron.bias -= L * d_hidden[i]

                gradients[l] = d_hidden

    def predict(self, x):
        x = np.array(x)
        output = self.forward_pass(x)
        return output