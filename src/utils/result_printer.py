import numpy as np

def print_final_results(accuracys, et, percent):
    print("-----------------------------------")
    print(f"Accuracy at {int(percent * 100)}% of training data:\nMean: {np.mean(accuracys)} Std Deviation: {np.std(accuracys)}")
    print(f"Execution time: {et:.3f} seconds")
    print("-----------------------------------")

def print_results(mode, percent, learning_rate, correct_count, incorrect_count, count_n, cost=None, pytorch=False, perceptron=False):
    print(f"{"PyTorch " if pytorch else ""}{"Digit" if mode == "digit" else "Face"} {"Neural Network" if not perceptron else "Perceptron"}")
    print(f"Trained with {int(percent * 100)}% of training data, learning rate of {learning_rate}")
    if cost: print(f"Final cost: {cost}")
    print(f"Correct: {correct_count}, Incorrect: {incorrect_count} | {correct_count / count_n * 100}%")