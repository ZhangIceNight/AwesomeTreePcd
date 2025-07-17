import torch
import torch.nn as nn
import torch.nn.functional as F





class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class TransformerSA(nn.Module):
    """
    Self Attention module based on Transformer.
    """

    def __init__(self, input_dim, num_heads):
        super(TransformerSA, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Input:
            x: [B, N, input_dim]
        
        Output:
            x: [B, N, input_dim]
        """
        B, N, _ = x.size()

        # Project inputs to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Rearrange for multi-head attention
        query = query.view(B, N, self.input_dim)
        key = key.view(B, N, self.input_dim)
        value = value.view(B, N, self.input_dim)

        # Apply multi-head attention    
        attn_output, _ = self.multihead_attn(query, key, value) # [B, N, input_dim]

        # Apply dropout, residual connection, and layer normalization
        x = self.dropout(attn_output)
        x = self.norm(x + x)

        return x


class NaivePCT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = TransformerSA(128, num_heads=8)
        self.sa2 = TransformerSA(128, num_heads=8)
        self.sa3 = TransformerSA(128, num_heads=8)
        self.sa4 = TransformerSA(128, num_heads=8)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1152, kernel_size=1, bias=False),
            nn.BatchNorm1d(1152),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(-1, -2)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=-1)

        x = x.transpose(-1, -2)
        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


if __name__ == '__main__':
    pc = torch.rand(4, 3, 1024).to('cuda')
    cls_label = torch.rand(4, 16).to('cuda')

    # testing for cls networks
    naive_pct_cls = NaivePCTCls().to('cuda')
    print(naive_pct_cls(pc).size())

    # testing for segmentation networks
    naive_pct_seg = NaivePCTSeg().to('cuda')
    print(naive_pct_seg(pc, cls_label).size())

 