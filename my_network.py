import torch
import torch.nn as nn


# Defining the CNN Model extending the pytorch NN Model
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Does not specify the input size for height and width;
        # the input must be a tensor of shape (batch_size, 3, height, width), where the height and width can be any valid dimension.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # First Step Value
        x = self.conv1(x)
        # Take the activation layer
        x = torch.relu(x)
        # Take the pooling layer
        x = self.pool(x)

        # Second Step
        # ConvNet
        x = self.conv2(x)
        # Activation
        x = torch.relu(x)
        # Pooling Layer
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)

        # Taking into the fully-connected layer
        x = self.fc1(x)
        # Activation
        x = torch.relu(x)

        # Take the second FC Layer short hand
        x = torch.relu(self.fc2(x))

        # Take the final fc layer which activated to the class
        x = self.fc3(x)
        return x