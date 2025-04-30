from perceptron.digit_perceptron import DigitPerceptron
import utils.data_parser as data_parser

if __name__ == "__main__":
    perceptron = DigitPerceptron(verbose=True, epochs=20, learning_rate=0.1)
    training_nums = data_parser.parse_numbers("trainingimages")
    training_data = data_parser.attach_labels(training_nums, "traininglabels", True)

    perceptron.train(training_data)

    test_data = data_parser.parse_numbers("testimages")
    test_labels = data_parser.get_labels("testlabels", True)

    guesses = perceptron.classify(test_data)

    correct_count = 0
    incorrect_count = 0
    for i in range(len(guesses)):
        # print(f"Guess: {guesses[i]}, Answer: {test_labels[i]}")
        if int(guesses[i]) == int(test_labels[i]):
            correct_count += 1
        else:
            incorrect_count += 1

    print(f"Correct: {correct_count}, Incorrect: {incorrect_count} | {correct_count / len(guesses) * 100}%")