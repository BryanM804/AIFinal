import numpy as np

def activation(z):
    return 1 if z > 0 else 0

class FacePerceptron():

    def __init__(self, epochs=10, learning_rate=0.1, verbose=False):
        self.weights = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

    # training_data should be tuples, the second element being the label
    def train(self, training_data):
        # self.weights = np.random.randn(len(self.biases), len(training_data[0][0])) * 0.05
        self.weights = np.zeros(len(training_data[0][0]) + 1)

        for i in range(self.epochs):
            if self.verbose: 
                print(f"Epoch: {i + 1}")
                correct_count = 0
                incorrect_count = 0

            weights_update = np.zeros_like(self.weights)

            for x in range(len(training_data)):
                Y = training_data[x][1]
                training_x = np.array(training_data[x][0], dtype=np.float32)
                X = np.append(training_x, 1) # add bias

                score = np.dot(self.weights, X)
                y_predict = activation(score)
            
                if y_predict != Y:
                    if self.verbose: incorrect_count += 1
                    weights_update += self.learning_rate * (Y - y_predict) * X
                elif self.verbose:
                    correct_count += 1
            self.weights += (weights_update / len(training_data))

            if self.verbose: print(f"Correct: {correct_count}, Incorrect: {incorrect_count}")

        if self.verbose:
            print(f"Ending Weights: {self.weights}")

    # Returns an array of guesses, test_data should not include labels
    def classify(self, test_data):
        guesses = []

        for x in test_data:
            npx = np.array(x, dtype=np.float32)
            npx_aug = np.append(npx, 1)

            score = np.dot(self.weights, npx_aug)
            guess = activation(score)

            guesses.append(guess)

        return guesses