from perceptron.digit_perceptron import DigitPerceptron
from perceptron.face_perceptron import FacePerceptron
import utils.data_parser as data_parser
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="digit",  help="Digit or face perceptron test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the perceptron with")
    parser.add_argument("--verbose", type=bool, default=False, help="Log extra information")
    args = parser.parse_args()

    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]

    for percent in percentages:
        if args.mode == "digit":
            perceptron = DigitPerceptron(verbose=args.verbose, epochs=args.epochs, learning_rate=0.1)
            training_nums = data_parser.parse_numbers("trainingimages")
            training_data = data_parser.attach_labels(training_nums, "traininglabels", True)
        else:
            perceptron = FacePerceptron(verbose=args.verbose, epochs=args.epochs)
            training_faces = data_parser.parse_faces("facedatatrain")
            training_data = data_parser.attach_labels(training_faces, "facedatatrainlabels", False)

        training_data = training_data[:int(len(training_data) * percent)]

        perceptron.train(training_data)

        if args.mode == "digit":
            test_data = data_parser.parse_numbers("testimages")
            test_labels = data_parser.get_labels("testlabels", True)
        else:
            test_data = data_parser.parse_faces("facedatatest")
            test_labels = data_parser.get_labels("facedatatestlabels", False)

        guesses = perceptron.classify(test_data)

        correct_count = 0
        incorrect_count = 0
        for i in range(len(guesses)):
            # print(f"Guess: {guesses[i]}, Answer: {test_labels[i]}")
            if int(guesses[i]) == int(test_labels[i]):
                correct_count += 1
            else:
                incorrect_count += 1

        print(f"{"Digit" if args.mode == "digit" else "Face"} Perceptron\n------------------\n")
        print(f"Trained with {int(percent * 100)}% of training data.")
        print(f"Correct: {correct_count}, Incorrect: {incorrect_count} | {correct_count / len(guesses) * 100}%")