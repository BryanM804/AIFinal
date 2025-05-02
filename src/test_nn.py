from three_layer.digit_nn import DigitNeuralNet
import utils.data_parser as data_parser
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="digit",  help="Digit or face perceptron test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the perceptron with")
    parser.add_argument("--verbose", type=bool, default=False, help="Log extra information")
    parser.add_argument("--learningrate", type=float, default=0.1, help="Learining rate of model, typically small like 0.1")
    args = parser.parse_args()

    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
    # percentages = [1]

    if args.mode == "digit":
        training_nums = data_parser.parse_numbers("trainingimages")
        training_labels = data_parser.get_labels("traininglabels", True)
    else:
        pass

    for percent in percentages:
        if args.mode == "digit":
            training_n = training_nums[:int(len(training_nums) * percent)]
            training_l = training_labels[:int(len(training_labels) * percent)]
            net = DigitNeuralNet(layer_size=len(training_nums[0]), verbose=args.verbose, epochs=args.epochs, learning_rate=args.learningrate)        
        else:
            pass

        cost = net.train(training_n, training_l)

        if args.mode == "digit":
            test_data = data_parser.parse_numbers("testimages")
            test_labels = data_parser.get_labels("testlabels", True)
        else:
            test_data = data_parser.parse_faces("facedatatest")
            test_labels = data_parser.get_labels("facedatatestlabels", False)

        predictions = net.classify(test_data)
        guesses = np.argmax(predictions, axis=1)

        test_labels = np.array(test_labels, dtype=np.int64)

        correct_count = 0
        incorrect_count = 0

        for i in range(len(guesses)):
            if args.verbose: 
                print(f"Guess: {guesses[i]}, Answer: {test_labels[i]}")
            if guesses[i] == test_labels[i]:
                correct_count += 1
            else:
                incorrect_count += 1

        print(f"{"Digit" if args.mode == "digit" else "Face"} Neural Network\n------------------")
        print(f"Trained with {int(percent * 100)}% of training data, learning rate of {args.learningrate}")
        print(f"Final cost: {cost}")
        print(f"Correct: {correct_count}, Incorrect: {incorrect_count} | {correct_count / len(guesses) * 100}%")