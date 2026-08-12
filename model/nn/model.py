import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block(nn.Module):
    """Squeeze-and-Excitation Block: Learns which features matter."""
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConvolution, self).__init__()
        self.A = nn.Parameter(A.clone().detach().to(torch.float32), requires_grad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        n, c, t, v = x.size()
        x = torch.einsum('nctv,vw->nctw', (x, self.A))
        return x

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0): # Added dropout arg
        super(STGCN_Block, self).__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout) # Use dynamic dropout
        )
        # ATTENTION UPGRADE
        self.se = SE_Block(out_channels)

    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.se(x) # Apply Attention
        return x

class STGCN(nn.Module):
    def __init__(self, num_classes=3, in_channels=6, t_kernel_size=9, hop_size=1, dropout=0.0): # Added dropout arg
        # Note: in_channels=6 (3 Pos + 3 Vel)
        super(STGCN, self).__init__()
        
        self.A = self.get_adjacency_matrix(17)
        self.data_bn = nn.BatchNorm1d(in_channels * 17)
        
        self.layers = nn.ModuleList([
            STGCN_Block(in_channels, 64, self.A, dropout=dropout),
            STGCN_Block(64, 64, self.A, dropout=dropout),
            STGCN_Block(64, 128, self.A, stride=2, dropout=dropout),
            STGCN_Block(128, 128, self.A, dropout=dropout),
            STGCN_Block(128, 256, self.A, stride=2, dropout=dropout),
            STGCN_Block(256, 256, self.A, dropout=dropout)
        ])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: (N, C, T, V)
        N, C, T, V = x.size()
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for layer in self.layers:
            x = layer(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)
        return self.fc(x)

    def get_adjacency_matrix(self, num_nodes):
        edges = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),  # head → shoulders (bridge fix)
                 (5,7),(7,9),(6,8),(8,10),(5,6),        # arms
                 (5,11),(6,12),(11,12),                 # torso
                 (11,13),(13,15),(12,14),(14,16)]       # legs
        A = torch.zeros(num_nodes, num_nodes)
        for i, j in edges: A[i, j] = A[j, i] = 1
        for i in range(num_nodes): A[i, i] = 1
        D = torch.sum(A, dim=1)
        D_inv = torch.pow(D, -0.5)
        D_inv[torch.isinf(D_inv)] = 0
        D_mat = torch.diag(D_inv)
        return torch.mm(torch.mm(D_mat, A), D_mat)