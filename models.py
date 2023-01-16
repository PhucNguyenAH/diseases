import torch
import torch.nn as nn
import torch.nn.functional as F

class Bin(nn.Module):
    num_features = 8630
    n_hidden_1 = 8192
    num_classes = 2
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.num_features, self.n_hidden_1)
        self.layer_out = nn.Linear(self.n_hidden_1, self.num_classes)

        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(self.n_hidden_1)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.fc1(x)))

        x = F.log_softmax(self.layer_out(x), -1)
        return x

class Five(nn.Module):
    num_features = 8630
    n_hidden_1 = 8192
    n_hidden_2 = 1024
    n_hidden_3 = 128
    num_classes = 5
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.num_features, self.n_hidden_1)
        self.fc2 = nn.Linear(self.n_hidden_1, self.n_hidden_2)
        self.fc3 = nn.Linear(self.n_hidden_2, self.n_hidden_3)
        self.layer_out = nn.Linear(self.n_hidden_3, self.num_classes)

        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(self.n_hidden_1)
        self.batchnorm2 = nn.BatchNorm1d(self.n_hidden_2)
        self.batchnorm3 = nn.BatchNorm1d(self.n_hidden_3)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(F.relu(self.batchnorm2(self.fc2(x))))
        x = self.dropout(F.relu(self.batchnorm3(self.fc3(x))))

        x = F.log_softmax(self.layer_out(x), -1)
        return x

class Six(nn.Module):
    num_features = 8630
    n_hidden_1 = 8192
    n_hidden_2 = 2048
    n_hidden_3 = 128
    num_classes = 6
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.num_features, self.n_hidden_1)
        self.fc2 = nn.Linear(self.n_hidden_1, self.n_hidden_2)
        self.fc3 = nn.Linear(self.n_hidden_2, self.n_hidden_3)
        self.layer_out = nn.Linear(self.n_hidden_3, self.num_classes)

        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(self.n_hidden_1)
        self.batchnorm2 = nn.BatchNorm1d(self.n_hidden_2)
        self.batchnorm3 = nn.BatchNorm1d(self.n_hidden_3)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(F.relu(self.batchnorm2(self.fc2(x))))
        x = self.dropout(F.relu(self.batchnorm3(self.fc3(x))))

        x = F.log_softmax(self.layer_out(x), -1)
        return x