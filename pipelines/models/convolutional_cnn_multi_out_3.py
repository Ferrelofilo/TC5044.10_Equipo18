import torch
import torch.nn as nn

class ConvolutionalSimpleModel(nn.Module):
    def __init__(self, input_len=10, out_features1=64, out_features2=32, kernel_size=3, padding=1):
        super(ConvolutionalSimpleModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_features1, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=out_features1, out_channels=out_features2, kernel_size=kernel_size, padding=padding)
        self.flatten = nn.Flatten()

        # Compute the flattened dimension
        self.flat_dim = self._calculate_flat_dim(input_len, in_channels, kernel_size, padding)
        
        # Define linear layers for each output
        self.y1_output = nn.Linear(out_features2 * input_len, 1)  # Adjusting input size
        self.y2_output = nn.Linear(out_features2 * input_len, 1)
        self.y3_output = nn.Linear(out_features2 * input_len, 1)

    def _calculate_flat_dim(self, input_len, kernel_size, padding):
        L_out1 = input_len + 2 * padding - kernel_size + 1
        L_out2 = L_out1 + 2 * padding - kernel_size + 1
        flat_dim = L_out2 * self.conv2.out_channels
        return flat_dim

    def forward(self, x):
        x = x.view(x.size(0),1,-1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        # Generate outputs for each target
        y1 = self.y1_output(x)
        y2 = self.y2_output(x)
        y3 = self.y3_output(x)
        
        return y1,y2,y3