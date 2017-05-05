import numpy as np
from numpy import dot, exp

class NeuronLayer():
	def __init__(self, num_neurons, num_inputs_per_neuron):
		self.synaptic_weights = 2 * np.random.random((num_inputs_per_neuron, num_neurons)) - 1

class NeuralNetwork():
	def __init__(self, num_neurons, num_inputs):
		self.num_layers = num_neurons.size
		self.layers = np.empty(self.num_layers, dtype=object)
		self.layers[0] = NeuronLayer(num_neurons[0], num_inputs)
		for i in range(1, self.num_layers):
			self.layers[i] = NeuronLayer(num_neurons[i], num_neurons[i-1])

	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_set_inputs, training_set_outputs, num_training_iterations):
		for iteration in range(num_training_iterations):
			output_from_layers = self.think(training_set_inputs)

			layer_errors = np.empty(self.num_layers, dtype=object)
			layer_deltas = np.empty(self.num_layers, dtype=object)
			layer_adjustments = np.empty(self.num_layers, dtype=object)
			
			layer_errors[-1] = training_set_outputs - output_from_layers[-1]
			layer_deltas[-1] = layer_errors[-1] * self.__sigmoid_derivative(output_from_layers[-1])
			for i in range(2, self.num_layers+1):
				layer_errors[-i] = layer_deltas[-i+1].dot(self.layers[-i+1].synaptic_weights.T)
				layer_deltas[-i] = layer_errors[-i] * self.__sigmoid_derivative(output_from_layers[-i])

			layer_adjustments[0] = training_set_inputs.T.dot(layer_deltas[0])
			self.layers[0].synaptic_weights += layer_adjustments[0]
			for i in range(1, self.num_layers):
				layer_adjustments[i] = output_from_layers[i-1].T.dot(layer_deltas[i])
				self.layers[i].synaptic_weights += layer_adjustments[i] * 0.1

	def think(self, inputs):
		output_from_layers = np.empty(self.num_layers, dtype=object)
		output_from_layers[0] = self.__sigmoid(dot(inputs, self.layers[0].synaptic_weights))
		for i in range(1, self.num_layers):
			output_from_layers[i] = self.__sigmoid(dot(output_from_layers[i-1], self.layers[i].synaptic_weights))
		return output_from_layers
		
	def print_weights(self, n):
		print(self.layers[n].synaptic_weights)
		
#Seed the random number generator
np.random.seed(12)

num_neurons = np.array([4, 3, 1])
num_inputs = 2

neural_network = NeuralNetwork(num_neurons, num_inputs)

#print("Stage 1) Random starting synaptic weights: ")
#neural_network.print_weights()

# The training set. We have 7 examples, each consisting of 3 input values
# and 1 output value.
#training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
#training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

# Train the neural network using the training set.
# Do it 60,000 times and make small adjustments each time.
#neural_network.train(training_set_inputs, training_set_outputs, 100)

#print("Stage 2) New synaptic weights after training: ")
#neural_network.print_weights()

data = np.genfromtxt("data1.csv", delimiter=",")
x, y, target = np.hsplit(data, 3)
xandy = np.concatenate((x,y),axis=1)
training_set_inputs = xandy
training_set_outputs = target

neural_network.train(training_set_inputs, training_set_outputs, 10000)

# Test the neural network with a new situation.
#print("Stage 3) Considering a new situation [1, 1, 0] -> ?: ")
outputs = neural_network.think(np.array([-1.3, -3.8]))
print(outputs[-1])