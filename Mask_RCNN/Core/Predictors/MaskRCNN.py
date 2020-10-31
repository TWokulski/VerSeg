from torch import nn
from collections import OrderedDict


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels,
                 layers,
                 dim_reduced,
                 number_of_classes):

        dictionary = OrderedDict()
        next_feature = in_channels

        for layer_idx, layer_features in enumerate(layers, 1):
            dictionary['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            dictionary['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        dictionary['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        dictionary['relu5'] = nn.ReLU(inplace=True)
        dictionary['mask_fcn_logits'] = nn.Conv2d(dim_reduced, number_of_classes, 1, 1, 0)
        super().__init__(dictionary)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
