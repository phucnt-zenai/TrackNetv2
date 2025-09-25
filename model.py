import torch
import torch.nn as nn

class Conv2DBlock(nn.Module):
    """ Conv + ReLU + BN"""
    def __init__(self, in_dim, out_dim, kernel_size, padding='same', bias=True, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """ Conv2DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))
        self.conv_3 = Conv2DBlock(out_dim, out_dim, (3, 3))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class TrackNetV2(nn.Module):
    """ Original structure but less two layers 
        Total params: 10,161,411
        Trainable params: 10,153,859
        Non-trainable params: 7,552
    """
    def __init__(self, in_dim=9, out_dim=3):
        super(TrackNetV2, self).__init__()
        self.down_block_1 = Double2DConv(in_dim=in_dim, out_dim=64)
        self.down_block_2 = Double2DConv(in_dim=64, out_dim=128)
        self.down_block_3 = Double2DConv(in_dim=128, out_dim=256)
        self.bottleneck = Triple2DConv(in_dim=256, out_dim=512)
        self.up_block_1 = Double2DConv(in_dim=768, out_dim=256)
        self.up_block_2 = Double2DConv(in_dim=384, out_dim=128)
        self.up_block_3 = Double2DConv(in_dim=192, out_dim=64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ model input shape: (F*3, 288, 512), output shape: (F, 288, 512) -> 3in - 3out """
        x1 = self.down_block_1(x)                                   # (64, 288, 512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                 # (64, 144, 256)
        x2 = self.down_block_2(x)                                   # (128, 144, 256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                 # (128, 72, 128)
        x3 = self.down_block_3(x)                                   # (256, 72, 128), one less conv layer
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                 # (256, 36, 64)
        x = self.bottleneck(x)                                      # (512, 36, 64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)  # (768, 72, 128)
        x = self.up_block_1(x)                                      # (256, 72, 128), one less conv layer
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)  # (384, 144, 256)
        x = self.up_block_2(x)                                      # (128, 144, 256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)  # (192, 288, 512)
        x = self.up_block_3(x)                                      # (64, 288, 512)
        x = self.predictor(x)                                       # (3, 288, 512)
        x = self.sigmoid(x)
        return  x

# SE Attention Block
class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Squeeze operation: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation operation: Fully connected layers
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Squeeze step
        squeeze = self.avg_pool(x).view(batch_size, channels)
        
        # Excitation step
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch_size, channels, 1, 1)
        
        # Scale input feature maps
        return x * excitation

# Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Query, Key, and Value
        query = self.query_conv(x).view(batch_size, -1, height * width)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        # Attention map
        attention = torch.bmm(query.transpose(1, 2), key)
        attention = torch.softmax(attention, dim=-1)

        # Weighted sum of values
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(batch_size, channels, height, width)
        
        # Return the weighted sum, scaled by gamma
        return self.gamma * out + x

# TrackNetV2 Model with SE and Self-Attention
class TrackNetV2_2(nn.Module):
    def __init__(self, in_dim=9, out_dim=3):
        super(TrackNetV2_2, self).__init__()
        self.down_block_1 = Double2DConv(in_dim=in_dim, out_dim=64)
        self.se1 = SEAttention(64)
        self.down_block_2 = Double2DConv(in_dim=64, out_dim=128)
        self.se2 = SEAttention(128)
        self.down_block_3 = Double2DConv(in_dim=128, out_dim=256)
        self.se3 = SEAttention(256)
        
        self.bottleneck = Triple2DConv(in_dim=256, out_dim=512)
        self.self_attention = SelfAttention(512)  # Apply self-attention to bottleneck
        
        self.up_block_1 = Double2DConv(in_dim=768, out_dim=256)
        self.se4 = SEAttention(256)
        self.up_block_2 = Double2DConv(in_dim=384, out_dim=128)
        self.se5 = SEAttention(128)
        self.up_block_3 = Double2DConv(in_dim=192, out_dim=64)
        self.se6 = SEAttention(64)
        
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)
        x1 = self.se1(x1)  # SE attention after first block
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)
        
        x2 = self.down_block_2(x)
        x2 = self.se2(x2)  # SE attention after second block
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)
        
        x3 = self.down_block_3(x)
        x3 = self.se3(x3)  # SE attention after third block
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)
        
        x = self.bottleneck(x)
        x = self.self_attention(x)  # Self-attention on bottleneck feature map
        
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
        x = self.up_block_1(x)
        x = self.se4(x)  # SE attention after first up block
        
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
        x = self.up_block_2(x)
        x = self.se5(x)  # SE attention after second up block
        
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
        x = self.up_block_3(x)
        x = self.se6(x)  # SE attention after third up block
        
        x = self.predictor(x)
        x = self.sigmoid(x)
        return x
