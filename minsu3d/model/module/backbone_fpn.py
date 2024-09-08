import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnext import PointNeXt

# B2

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, 1)
        self.conv2 = nn.Conv1d(out_channels // 4, out_channels // 4, 3, padding=1)
        self.conv3 = nn.Conv1d(out_channels // 4, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_channels // 4)
        self.bn2 = nn.BatchNorm1d(out_channels // 4)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += residual
        return self.relu(out)

class EnhancedFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = BottleneckBlock(in_channels, out_channels)
        self.conv2 = BottleneckBlock(out_channels, out_channels)
        self.conv3 = BottleneckBlock(out_channels, out_channels)
        
        self.top_down2 = nn.Conv1d(out_channels, out_channels, 1)
        self.top_down1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        
        p3 = c3
        p2 = c2 + self.top_down2(F.interpolate(p3, size=c2.shape[-1], mode='linear'))
        p1 = c1 + self.top_down1(F.interpolate(p2, size=c1.shape[-1], mode='linear'))
        
        return torch.cat([p1, p2, p3], dim=1)

class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.relu(self.norm(self.fc1(x)))
        x = self.fc2(x)
        return x

class BackboneFPN(nn.Module):
    def __init__(self, input_channel, output_channel, block_channels, block_reps, sem_classes):
        super().__init__()

        self.pointnext = PointNeXt()

        fpn_channels = output_channel // 2

        self.pre_fpn_conv = BottleneckBlock(output_channel, output_channel)
        self.fpn = EnhancedFPN(output_channel, fpn_channels)

        post_fpn_channels = fpn_channels * 3

        self.post_fpn_conv = BottleneckBlock(post_fpn_channels, output_channel)

        self.semantic_branch = EnhancedMLP(output_channel, output_channel // 2, sem_classes)

        self.offset_branch = EnhancedMLP(output_channel, output_channel // 2, 3)

    def forward(self, input_dict, model_ids=None):
        output_dict = {}
        point_features_dense = self.pointnext(input_dict)
        
        x = self.pre_fpn_conv(point_features_dense)
        fpn_features = self.fpn(x)
        enhanced_features = self.post_fpn_conv(fpn_features)

        point_features_sparse = enhanced_features.transpose(1, 2).reshape(-1, enhanced_features.shape[1])
        
        output_dict["point_features"] = point_features_sparse
        output_dict["semantic_scores"] = self.semantic_branch(point_features_sparse)
        output_dict["point_offsets"] = self.offset_branch(point_features_sparse)
        
        return output_dict
