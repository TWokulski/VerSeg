from torch import nn
import torch.nn.functional as F


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_prediction = nn.Conv2d(in_channels, 4 * num_anchors, 1)

        for layer in self.children():
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_prediction(x)
        return logits, bbox_reg
