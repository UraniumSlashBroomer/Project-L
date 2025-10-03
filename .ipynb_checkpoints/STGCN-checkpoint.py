import torch
from torch import nn
import mediapipe as mp

mp_hands = mp.solutions.hands

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, A):
        super().__init__()
        self.A = A
        self.D = self.A.sum(dim=1)
        self.D = torch.diag(torch.pow(self.D, -1/2))
        self.A = torch.matmul(self.D, self.A)
        self.A = torch.matmul(self.A, self.D)
        self.W = nn.Parameter(torch.empty(in_features, out_features))

        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        """
        A = adjacent matrix [N, N]
        x = [Batch, C_in, T, N]
        W = learnable parameters [C_in, C_out]
        """
        x = torch.einsum('nm,bctm->bctn', self.A, x) # [N, N] [B, C_in, T, N] -> [B, C_in, T, N]
        x = torch.einsum('bctn,cv->bvtn', x, self.W) #  [B, C_in, T, N] [C_in, C_out] -> [B, C_out, T, N]
        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=(9, 1), stride=1):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        padding = (kernel_size[0] - 1) // 2
        if kernel_size[1] != 1:
            final_padding = (padding, padding)
        else:
            final_padding = (padding, 0)
    
        self.tcn = nn.Conv2d(in_channels=out_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=(stride, 1),
                             padding=final_padding)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class SpatialTemporalGraphConvNetwork(nn.Module):
    def __init__(self, num_classes, num_channels, num_nodes=21, threshold=0.5, device='cpu'):
        super().__init__()
        self.A = torch.eye(num_nodes, device=device)
        self.threshold = threshold # threshold for inference
        for i, j in mp_hands.HAND_CONNECTIONS:
            self.A[i, j] = 1
            self.A[j, i] = 1

        self.device = device
        self.block1 = STGCNBlock(num_channels, 64, A=self.A, kernel_size=(9, 1))
        self.block2 = STGCNBlock(64, 128, A=self.A, kernel_size=(9, 1))
        self.block3 = STGCNBlock(128, 196, A=self.A, kernel_size=(9, 1))

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # temporal and nodes -> [B, C, 1, 1]
        self.fc1 = nn.Linear(196, 2)

    def forward(self, x):
        """
        x [B, num_channels, T, N]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
