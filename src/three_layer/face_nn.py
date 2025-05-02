import numpy as np
from three_layer.nn_funcs import sigmoid, sigmoid_d, cost

class DigitNeuralNet():

    def __init__(self, layer_size, learning_rate=0.1, hidden_layers=2, epochs=10, verbose=False):
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.verbose = verbose

        self.layer_sizes = [layer_size]
        # input, hidden layers, output
        for l in range(self.hidden_layers):
            self.layer_sizes.append(layer_size)
        self.layer_sizes.append(10)
        pass
    
    def forward_propagation(self, training_set):
        a = np.array(training_set, dtype=np.float32)
        acts = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            zs.append(z)
            a = sigmoid(z)
            acts.append(a)
        return acts, zs
        

    def backpropagation(self, acts, zs, y):
        n = len(y)
        l_count = len(self.weights)
        # Using 2 separate delta arrays because it is easier for my brain to comprehend
        # Rather than rolling them and unrolling like the slideshow suggests
        weight_delta = [0] * l_count
        bias_delta = [0] * l_count

        # Compute output layer error
        err = acts[-1] - y
        weight_delta[-1] = np.dot(acts[-2].T, err) / n
        bias_delta[-1] = np.sum(err, keepdims=True) / n

        # Propagate error backwards through hidden layers
        # Start at last hidden layer, decrement until -1
        for l in range(l_count - 2, -1, -1):
            err = np.dot(err, self.weights[l + 1].T) * sigmoid_d(zs[l])
            weight_delta[l] = np.dot(acts[l].T, err) / n
            bias_delta[l] = np.sum(err, keepdims=True) / n

        # Everything is divided by n since the backpropagation is being done for all
        # of the elements of training data at once.
            
        return weight_delta, bias_delta


    def train(self, training_data, labels):
        # Convert integer labels into vectors for the output of the network
        yarr = np.array(labels, dtype=np.float32)
        y = np.eye(10)[yarr.astype(int)]
        print(y)
        # Make random arrays for weights
        # Make arrays for biases
        for i in range(len(self.layer_sizes) - 1):
            # Random matrix of size a, b
            # a = neurons in layer x
            # b = input size of layer x + 1
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))

        for epoch in range(self.epochs):
            acts, zs = self.forward_propagation(training_data)
            current_cost = cost(y, acts[-1]) # acts[-1] would be the output vector
            weight_d, bias_d = self.backpropagation(acts, zs, y)
            
            for i in range(len(self.weights)):
                self.weights[i] -= weight_d[i] * self.learning_rate
                self.biases[i] -= bias_d[i] * self.learning_rate

            if self.verbose:
                print(f"Epoch {epoch + 1}, Cost: {current_cost}")
        return

    
    def classify(self, test_data):
        acts, zs = self.forward_propagation(test_data)
        return acts[-1]