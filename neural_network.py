import numpy as np

class Neuron:
    def __init__(self, num_inputs, layer, activation='relu'):
        self.weights = np.random.randn(num_inputs) * np.sqrt(2 / num_inputs)
        self.bias = np.random.randn()
        self.layer = layer
        self.activation_type = activation

    def activation_function(self, z):
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_type == 'relu':
            return np.maximum(0, z)

    def derivative_function(self):
        if self.activation_type == 'sigmoid':
            return self.activation * (1 - self.activation)
        elif self.activation_type == 'relu':
            return np.where(self.activation > 0, 1, 0)
    
    def forward_pass(self, input_vector):
        self.inputs = input_vector
        self.z = np.dot(self.weights, input_vector) + self.bias
        self.activation = self.activation_function(self.z)

        return self.activation
    
class Layer:
    def __init__(self, num_neurons, inputs_per_neuron, layer, activation='sigmoid'):
        self.neurons = [Neuron(inputs_per_neuron, layer, activation) for _ in range(num_neurons)]

    def forward_pass(self, input_vector):
        output_vector = []

        for neuron in self.neurons:
            output_vector.append(neuron.forward_pass(input_vector))

        return np.array(output_vector)
    
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, num_neurons, num_inputs, activation='sigmoid'):
        layer_index = len(self.layers)
        layer = Layer(num_neurons, num_inputs, layer_index, activation)

        self.layers.append(layer)

    def forward_pass(self, input_vector):
        for i, layer in enumerate(self.layers):
            input_vector = layer.forward_pass(input_vector)
            if i == len(self.layers) - 1:
                input_vector = self.softmax(input_vector)
        return input_vector
    
    def loss_function(self, true_values, predicted_values):
        loss = -np.sum(true_values * np.log(predicted_values + 1e-8))

        return loss
    
    def train(self, x, y, L=0.1):
        output = self.forward_pass(x)

        gradients = [None] * len(self.layers)

        last_layer = self.layers[-1]
        d_output = last_layer.forward_pass(x) - y

        for i, neuron in enumerate(last_layer.neurons):
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
                    d_hidden[i] = neuron.derivative_function() * error_sum
                    neuron.weights -= L * d_hidden[i] * neuron.inputs
                    neuron.bias -= L * d_hidden[i]

                gradients[l] = d_hidden

    def predict_class(self, x):
        probs = self.forward_pass(x)
        return np.argmax(probs)
    
    def softmax(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum()