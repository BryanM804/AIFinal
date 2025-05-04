
import utils.data_parser as data_parser
import argparse
import numpy as np
import torch
from pytorch.pytorch_nn import PytorchNeuralNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="digit",  help="Digit or face perceptron test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the perceptron with")
    parser.add_argument("--verbose", type=bool, default=False, help="Log extra information")
    parser.add_argument("--learningrate", type=float, default=0.1, help="Learining rate of model, typically small like 0.1")
    parser.add_argument("--reps", type=int, default=5, help="Number of times to repeat for accuracy measurements")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Cuda availabe, using GPU.")
        device = torch.device("cuda")
    else:
        print("Cuda unavailable, using CPU")
        device = torch.device("cpu")

    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # percentages = [1]

    if args.mode == "digit":
        training_nums = data_parser.parse_numbers("trainingimages")
        training_data = data_parser.attach_labels(training_nums, "traininglabels", True)
    else:
        training_faces = data_parser.parse_faces("facedatatrain")
        training_data = data_parser.attach_labels(training_faces, "facedatatrainlabels", False)

    for percent in percentages:
        accuracys = []

        for i in range(args.reps):
            # Randomly take a percentage of the training data for each rep
            training_indices = np.random.choice(len(training_data), int(len(training_data) * percent), replace=False)
            training_sets = [training_data[i] for i in training_indices]
            training_x, training_y = zip(*training_sets)
            training_x = torch.from_numpy(np.array(training_x, dtype=np.float32))
            training_y = torch.from_numpy(np.array(training_y, dtype=np.float32)).long()
            # Create neural net
            
            net = PytorchNeuralNet(len(training_x[0]), learning_rate=args.learningrate, output_size=10, epochs=args.epochs, verbose=args.verbose)
            
            # Send to device (GPU if available)
            net.to(device)
            training_x = training_x.to(device)
            training_y = training_y.to(device)

            # Train network
            net.train(training_x, training_y)

            if args.mode == "digit":
                test_data = data_parser.parse_numbers("testimages")
                test_labels = data_parser.get_labels("testlabels", True)
            else:
                test_data = data_parser.parse_faces("facedatatest")
                test_labels = data_parser.get_labels("facedatatestlabels", False)
            
            test_data = torch.from_numpy(np.array(test_data, dtype=np.float32))
            test_data = test_data.to(device)

            # Run network
            
            prediction = net(test_data)
            guesses = [torch.argmax(x).item() for x in prediction]

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

            print(f"PyTorch {"Digit" if args.mode == "digit" else "Face"} Neural Network\n------------------")
            print(f"Trained with {int(percent * 100)}% of training data, learning rate of {args.learningrate}")
            print(f"Correct: {correct_count}, Incorrect: {incorrect_count} | {correct_count / len(guesses) * 100}%")
            accuracys.append(correct_count / len(guesses) * 100)
        print("-----------------------------------")
        print(f"Accuracy at {int(percent * 100)}% of training data:\n Mean: {np.mean(accuracys)} Std Deviation: {np.std(accuracys)}")
        print("-----------------------------------")