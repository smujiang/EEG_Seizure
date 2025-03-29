

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads=8, expansion_factor=4, conv_kernel_size=5, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        
        self.layer_norm2 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * expansion_factor, kernel_size=1),  # Expansion
            nn.GELU(),
            nn.Conv1d(dim * expansion_factor, dim, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=dim),  # Depthwise conv
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1)  # Projection
        )

        self.layer_norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Multi-head Self-Attention
        x = x + self.self_attn(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))[0]
        
        # Convolution Module (Depthwise separable)
        x = x + self.conv(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2)
        
        # Feedforward Network
        x = x + self.ff(self.layer_norm3(x))
        
        return x

class ConformerEEG(nn.Module):
    def __init__(self, input_channels=19, seq_length=6400, model_dim=128, num_layers=5, heads=8, conv_kernel_size=5):
        super(ConformerEEG, self).__init__()
        
        # Convolutional Subsampling to reduce sequence length
        self.conv_subsample = nn.Sequential(
            nn.Conv1d(input_channels, model_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        
        # Conformer Blocks
        self.conformer_blocks = nn.Sequential(*[
            ConformerBlock(dim=model_dim, heads=heads, conv_kernel_size=conv_kernel_size) for _ in range(num_layers)
        ])
        
        # Fully connected layer
        self.fc = nn.Linear(model_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_subsample(x)  # Reduce sequence length
        x = x.permute(0, 2, 1)  # (B, Channels, Seq) -> (B, Seq, Channels) for Conformer
        x = self.conformer_blocks(x)  # Pass through Conformer Blocks
        x = self.fc(x)  # Output layer
        x = self.sigmoid(x).squeeze(-1)  # (B, Seq, 1) -> (B, Seq)
        
        return x

if __name__ == "__main__":
    # Example input
    batch_size = 8
    input_tensor = torch.randn(batch_size, 19, 6400)  # EEG input (Batch, Channels, Sequence)

    # Model Initialization
    model = ConformerEEG()
    output = model(input_tensor)

    print(output.shape)  # Expected output: (8, 1600) due to downsampling

    print("Done")
