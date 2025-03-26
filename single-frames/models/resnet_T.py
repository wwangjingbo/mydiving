import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=11, output_dim=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.linear_score = nn.Linear(512 * block.expansion, output_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

class ResNet50_T(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=6, num_classes=11, checkpoint_chunk_size=4,num_frames=128, dropout_rate=0.1):
        super(ResNet50_T, self).__init__()
        self.feature_extractor = ResNet50()
        self.fc_proj = nn.Linear(2048, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames*49, d_model))
        init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout_rate)
        self.pre_ln = nn.LayerNorm(d_model)
        self.attn_pool = nn.Sequential(nn.Linear(d_model, 1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.classifier = nn.Linear(d_model, num_classes)
        self.regressor = nn.Linear(d_model, 1)
        self.checkpoint_chunk_size = checkpoint_chunk_size

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        feats = []
        for i in range(0, x_reshaped.size(0), self.checkpoint_chunk_size):
            x_chunk = x_reshaped[i: i + self.checkpoint_chunk_size]
            feat_chunk = checkpoint(self.feature_extractor, x_chunk)
            feat_chunk = F.adaptive_avg_pool2d(feat_chunk, (4, 4))
            feats.append(feat_chunk)
        feat = torch.cat(feats, dim=0)  

        _, C_feat, H_feat, W_feat = feat.shape
        feat = feat.view(B, T, C_feat, H_feat * W_feat)  
        feat = feat.permute(0, 1, 3, 2).reshape(B, T * H_feat * W_feat, C_feat)  

        features = self.fc_proj(feat)  
        features = features + self.pos_embed[:, :features.size(1), :]
        features = self.dropout(features)
        features = self.pre_ln(features)
        features = features.permute(1, 0, 2)

        transformer_out = self.transformer_encoder(features)
        transformer_out = transformer_out.permute(1, 0, 2)  # [B, N, D]

        attn_scores = F.softmax(self.attn_pool(transformer_out), dim=1)  # [B, N, 1]
        pooled = (attn_scores * transformer_out).sum(dim=1)  # [B, D]

        class_logits = self.classifier(pooled)
        regression_out = self.regressor(pooled)
        return class_logits, regression_out