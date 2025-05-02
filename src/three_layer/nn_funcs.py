import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_d(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_d(z):
    return (z > 0).astype(np.float32)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 1e-8 is very small, avoids taking a log of 0 in rare cases

# Categorical cross entropy for softmax/digits
def cost_CCE(y, computed_y):
    return -np.sum(y * np.log(computed_y + 1e-8)) / len(y)

# Binary Cross Entropy for sigmoid
def cost_BCE(y, computed_y):
    n = len(y)
    cost = -np.sum(y * np.log(computed_y + 1e-8) + (1 - y) * np.log(1 - computed_y + 1e-8)) / n
    return cost