import numpy as np

class DigitPerceptron():

    def __init__(self, epochs=10, classes=10, learning_rate=0.1, verbose=False):
        self.weights = []
        # self.biases = np.random.randn(classes) * 0.05
        self.biases = np.zeros(classes)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

    # training_data should be tuples, the second element being the label
    def train(self, training_data):
        # self.weights = np.random.randn(len(self.biases), len(training_data[0][0])) * 0.05
        self.weights = np.zeros((len(self.biases), len(training_data[0][0])))

        if self.verbose:
            print(f"Starting Weights: {self.weights}")
            print(f"Starting Biases: {self.biases}")

        for i in range(self.epochs):
            if self.verbose: 
                print(f"Epoch: {i + 1}")
                correct_count = 0
                incorrect_count = 0

            weights_update = np.zeros_like(self.weights)
            biases_update = np.zeros_like(self.biases)

            for x in range(len(training_data)):
                Y = training_data[x][1]
                training_x = np.array(training_data[x][0], dtype=np.float32)

                scores = np.dot(self.weights, training_x) + self.biases
                y_predict = np.argmax(scores)
            
                if y_predict != Y:
                    if self.verbose: incorrect_count += 1
                    weights_update[Y] += self.learning_rate * training_x
                    biases_update[Y] += self.learning_rate
                    weights_update[y_predict] -= self.learning_rate * training_x
                    biases_update[y_predict] -= self.learning_rate
                elif self.verbose:
                    correct_count += 1
            self.weights += weights_update
            self.biases += biases_update

            if self.verbose: print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")

        if self.verbose:
            print(f"Ending Weights: {self.weights}")
            print(f"Ending Biases: {self.biases}")

    # Returns an array of guesses, test_data should not include labels
    def classify(self, test_data):
        guesses = []

        for x in test_data:
            npx = np.array(x, dtype=np.float32)

            scores = np.dot(self.weights, npx) + self.biases
            guess = np.argmax(scores)

            guesses.append(guess)

        return guesses