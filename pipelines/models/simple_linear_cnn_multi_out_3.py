import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import mean_absolute_error, r2_score


class SimpleLinearCnnMO3(nn.Module):
    def __init__(self, input_len=10, out_features1=64, out_features2=32, bias=True):
        super(SimpleLinearCnnMO3, self).__init__()
        self.first_dense = nn.Linear(input_len, out_features1, bias=bias)
        self.second_dense = nn.Linear(out_features1, out_features2, bias=bias)
        self.y1_output = nn.Linear(out_features2, 1)
        self.y2_output = nn.Linear(out_features2, 1)
        self.y3_output = nn.Linear(out_features2, 1)

    def forward(self, x):
        x = torch.relu(self.first_dense(x))
        x = torch.relu(self.second_dense(x))
        y1 = self.y1_output(x)  # common_flares
        y2 = self.y2_output(x)  # moderate_flares
        y3 = self.y3_output(x)  # severe_flares
        return y1, y2, y3

class ConvolutionalSimpleModel(nn.Module):
    def __init__(self, input_len=10, out_features1=64, out_features2=32, kernel_size=3, padding=1):
        super(ConvolutionalSimpleModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_features1, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=out_features1, out_channels=out_features2, kernel_size=kernel_size, padding=padding)
        #Could add batch normalization
        self.flatten = nn.Flatten()
        self.y1_output = nn.Linear(out_features2 * input_len, 1)  # Adjusting input size
        self.y2_output = nn.Linear(out_features2 * input_len, 1)
        self.y3_output = nn.Linear(out_features2 * input_len, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        y1 = self.y1_output(x)  # common_flares
        y2 = self.y2_output(x)  # moderate_flares
        y3 = self.y3_output(x)  # severe_flares
        return y1, y2, y3


