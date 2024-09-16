import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class simulated_data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Define a class for a 1D Convolutional Neural Network
class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, dtype=torch.float, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x
    
    
class CNN_simple(nn.Module):
    def __init__(
            self,
            num_classes, 
            ):
        super().__init__()
        conv_block = BasicConv1d
        self.num_unique_classes = num_classes

        self.conv1 = conv_block(1, 256, kernel_size=5, stride=1, padding=2)

        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = conv_block(256, 512, kernel_size=3, stride=1, padding=1)

        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)



        self.fc1 = nn.LazyLinear(2000)
        self.fc2 = nn.LazyLinear(1000)
        self.fc3 = nn.LazyLinear(num_classes)

    # define the forward method that applies the convolutional and fully connected layers to the input data
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)

        x = self.maxpool_1(x)

        x = self.conv2(x)

        x = self.maxpool_2(x)

        x = F.dropout1d(x, p=0.2)

        # Fully connected Linear
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    


class CNN_bragg(nn.Module):
    def __init__(
            self,
            num_classes, 
            ):
        super().__init__()
        conv_block = BasicConv1d
        self.num_unique_classes = num_classes

        self.conv1 = conv_block(1, 256, kernel_size=31, stride=1, padding=15)

        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = conv_block(256, 512, kernel_size=31, stride=1, padding=15)

        self.maxpool_2 = nn.MaxPool1d(kernel_size=3, stride=2)



        # self.fc1 = nn.LazyLinear(2000)
        self.fc2 = nn.LazyLinear(1000)
        self.fc3 = nn.LazyLinear(num_classes)

    # define the forward method that applies the convolutional and fully connected layers to the input data
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)

        x = self.maxpool_1(x)

        x = self.conv2(x)

        x = self.maxpool_2(x)

        x = F.dropout1d(x, p=0.2)

        # Fully connected Linear
        x = torch.flatten(x, 1)
        # x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x