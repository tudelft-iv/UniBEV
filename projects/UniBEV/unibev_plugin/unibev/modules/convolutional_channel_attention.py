import torch
import torch.nn as nn

class ConvChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ConvChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class MultiModalConvChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(MultiModalConvChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_feats, pts_feats):
        out = []
        if img_feats is not None:
            img_avg_out = self.shared_mlp(self.avg_pool(img_feats))
            img_max_out = self.shared_mlp(self.max_pool(img_feats))
            img_out = img_avg_out + img_max_out
            out.append(img_out.squeeze(-1))
        if pts_feats is not None:
            pts_avg_out = self.shared_mlp(self.avg_pool(pts_feats))
            pts_max_out = self.shared_mlp(self.max_pool(pts_feats))
            pts_out = pts_avg_out + pts_max_out
            out.append(pts_out.squeeze(-1))

        out = torch.cat(out, -1)

        return self.softmax(out)

class ConvSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ConvSpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


