from torch import nn
from torchvision import models
from torchvision.ops import misc


class ResBackbone(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()

        resnet_body = models.resnet.__dict__[backbone_name](
            pretrained=True, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in resnet_body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.body = nn.ModuleDict(d for i, d in enumerate(resnet_body.named_children()) if i < 8)
        self.in_channels = 2048
        self.out_channels = 256

        self.inner_block_module = nn.Conv2d(self.in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for module in self.body.values():
            x = module(x)

        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x
