import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import Tensor


class EEG_Clip_ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=64, n_classes=2):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(5600,8),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(8, 8),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(8, n_classes)
        )

    def forward(self, x):
        # print(x.shape)
        # x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        # print(out.shape)
        return out

# use conv to capture local features, instead of postion embedding.
class EEG_Clip_Embedding(nn.Module):
    def __init__(self, emb_size=64):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (3, 20), stride=(1, 3)),
            nn.Conv2d(40, 16, (3, 20), stride=(1, 3)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(16, emb_size, (3, 3), stride=(2, 2)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class my_cnn_model(nn.Sequential):
    def __init__(self, emb_size=40, n_classes=2, **kwargs):
        super().__init__(

            EEG_Clip_Embedding(emb_size),
            EEG_Clip_ClassificationHead(emb_size, n_classes)
        )

if __name__ == "__main__":
    n_classes = 2

    input = torch.randn(8, 1, 19, 6400)
    # 1: input channel
    # 2. output dimention on the dimention index of input channel
    m = nn.Conv2d(1, 40, (3, 20), stride=(1, 3))
    # >>> # non-square kernels and unequal stride and with padding and dilation
    # >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    # >>> input = torch.randn(20, 16, 50, 100)
    output = m(input)
    print(output.shape)
    m = nn.Conv2d(40, 16, (3, 20), stride=(1, 3))
    output = m(output)
    print(output.shape)

    output = nn.BatchNorm2d(16)(output)
    print(output.shape)
    output = nn.ELU()(output)
    print(output.shape)
    output = nn.AvgPool2d((1, 75), (1, 15))(output)
    print(output.shape)
    output = nn.Conv2d(16, 64, (3, 3), stride=(2, 2)) (output)
    print(output.shape)

    output = nn.Flatten()(output)
    print(output.shape)
    output = nn.Linear(8960, 256)(output)
    print(output.shape)
    output = nn.ELU()(output)
    print(output.shape)
    output = nn.Dropout(0.5)(output)
    print(output.shape)
    output = nn.Linear(256, 32)(output)
    print(output.shape)
    output = nn.ELU()(output)
    print(output.shape)
    output = nn.Dropout(0.3)(output)
    print(output.shape)
    output = nn.Linear(32, n_classes)(output)
    print(output.shape)


    # input size : [batchsize, 1, 19, 6400]
    # 19 is related to the channels that can be used in prediction
    # 6400 is related to the length of the eeg clip
    # 




