import torch
import torch.nn as nn
import torch.optim as optim

class SeizureDetectionModel(nn.Module):
    def __init__(self, input_channels=19, seq_length=6400, hidden_dim=128, num_layers=2):
        super(SeizureDetectionModel, self).__init__()
        
        # Convolutional layers to extract spatial features
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM to capture temporal dependencies
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional LSTM
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        # x = self.pool(x)
        
        # Transpose for LSTM (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        
        # Fully connected layer for each time step
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x.squeeze(-1)  # Output shape: (batch, seq_length)

# Example usage
model = SeizureDetectionModel()
input_tensor = torch.randn(8, 19, 6400)  # Batch size 8
output = model(input_tensor)
print(output.shape)  # Expected output: (8, 6400)



print("DONE")

