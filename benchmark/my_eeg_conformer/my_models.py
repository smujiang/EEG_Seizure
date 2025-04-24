import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from einops import rearrange
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

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])



class my_conformer_model(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=2, **kwargs):
        super().__init__(

            EEG_Clip_Embedding(emb_size),
            TransformerEncoder(depth, emb_size),
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




