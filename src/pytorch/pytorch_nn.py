import torch
import torch.nn as nn
import numpy as np

class PytorchNeuralNet(nn.Module):
    
    def __init__(self, input_size, learning_rate=0.1, output_size=10, epochs=10, verbose=False):
        super(PytorchNeuralNet, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.verbose = verbose
        
        # Sequential to stack layers
        layers = [
            # Input number, neurons leaving them the same for now to match the
            # manual implementation
            nn.Linear(input_size, input_size), # Input -> Hidden 1
            nn.ReLU(),
            nn.Linear(input_size, input_size), # Hidden 1 -> Hidden 2
            nn.ReLU(),
            nn.Linear(input_size, output_size), # Output Layer
        ]
        if output_size < 2:
            layers.append(nn.Sigmoid())
            # Softmax is built into the CrossEntropyLoss function
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    def train(self, X, y):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        loss_calc = nn.BCELoss() if self.output_size < 2 else nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self(X)
            loss = loss_calc(output, y)
            loss.backward()
            optimizer.step()

            if self.verbose:
                print(f"Epoch {epoch + 1}, Cost: {loss}")