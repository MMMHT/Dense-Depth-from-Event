import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform import DeformBottleneck


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, leaky_relu=True):
        """StereoNet uses leaky relu (alpha = 0.2)"""
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=False) if leaky_relu else nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetFeature(nn.Module):
    def __init__(self, in_channels=3,
                 base_channels=32,
                 zero_init_residual=True,
                 groups=1,
                 width_per_group=64,
                 norm_layer=None):
        super(ResNetFeature, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        layers = [3, 4, 6]  # ResNet-40

        # self.inplanes = 64
        self.inplanes = in_channels
        self.dilation = 1

        self.groups = 1
        self.base_width = width_per_group

        stride = 1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=stride, groups = self.groups,
                                             padding=3, bias=False),
                                   nn.BatchNorm2d(self.inplanes),
                                   # nn.GroupNorm(self.groups, self.inplanes),
                                   nn.ReLU(inplace=False))

        self.layer1 = self._make_layer(Bottleneck, in_channels, layers[0])  # H/2
        self.layer2 = self._make_layer(Bottleneck, in_channels * 2, layers[1], stride=2)  # H/4
        self.layer3 = self._make_layer(DeformBottleneck, in_channels * 4, layers[2], stride=2)  # H/8

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                # nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,groups=self.groups),
                # nn.GroupNorm(self.groups, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        return [layer1, layer2, layer3]


#



class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=128,
                 num_levels=3):
        # FPN paper uses 256 out channels by default
        super(FeaturePyramidNetwork, self).__init__()

        assert isinstance(in_channels, list)

        self.in_channels = in_channels
        self.group = [1,1,1]
        # self.group = out_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()


        for i in range(num_levels):
            lateral_conv = nn.Conv2d(in_channels[i], out_channels[i], kernel_size=1, groups=self.group[i])

            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels[i], out_channels[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=False))

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # Inputs: resolution high -> low
        assert len(self.in_channels) == len(inputs)

        # Build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='bilinear')

        # Build outputs
        out = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return out


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.resnet_feature_network = ResNetFeature(in_channels=in_channels)
        self.feature_pyramid_network = FeaturePyramidNetwork(in_channels=[in_channels * 4, in_channels*8 , in_channels*16 ],
                                                             out_channels=[in_channels, in_channels , in_channels])

    def forward(self, x):
        resnet_feature = self.resnet_feature_network(x)
        # return resnet_feature
        feature = self.feature_pyramid_network(resnet_feature)
        return feature



class Handcrafted_Feature(nn.Module):
    def __init__(self, in_planes, kernel_size=3, num_levels=3):  # stride, padding, dilation, groups, bias, padding_mode
        super(Handcrafted_Feature, self).__init__()

        # initialize the kernel with the size 1 output, 1 input, 3x1x1
        self.kernel_3d = nn.Parameter(torch.zeros(1, 1, kernel_size, 3, 3))

        self.kernel_3d.data[:, :, 0, :, :] = -1
        self.kernel_3d.data[:, :, 1, :, :] = 2
        self.kernel_3d.data[:, :, 2, :, :] = -1

        self.kernel_size = kernel_size
        self.kernel_3d.requires_grad = False

        self.conv1 = nn.Sequential(nn.Conv3d(in_planes, in_planes, kernel_size=3, padding=1, stride=1, bias=False),
                                      nn.BatchNorm3d(in_planes),
                                        nn.ReLU(inplace=False),
                                   nn.Conv3d(in_planes, in_planes, kernel_size=3, padding=1, stride=1, bias=False),
                                        nn.BatchNorm3d(in_planes),
                                        nn.ReLU(inplace=False),
                                   )

        self.conv2 = nn.Sequential(nn.Conv3d(in_planes, in_planes, kernel_size=3, padding=1, stride=1, bias=False),
                                   nn.BatchNorm3d(in_planes),
                                   nn.ReLU(inplace=False),
                                   nn.Conv3d(in_planes, in_planes, kernel_size=3, padding=1, stride=1, bias=False),
                                      nn.BatchNorm3d(in_planes), )


        in_channels = [in_planes * 4, in_planes * 4, in_planes * 4]
        out_channels = [in_planes, in_planes, in_planes]

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(num_levels):
            lateral_conv = nn.Conv2d(in_channels[i], out_channels[i], 1)
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels[i], out_channels[i], 3, padding=1),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=False))

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):


        num_scales = len(x)
        Handcrafted_Feature_Pyramid = []

        for i in range(num_scales):
            output = x[i].unsqueeze(1)
            output_dx = nn.functional.conv3d(output, self.kernel_3d, stride=1, padding=(1, 1, 1))
            output_dxx = nn.functional.conv3d(output_dx, self.kernel_3d, stride=1, padding=(1, 1, 1))
            output_dx2 = output_dx ** 2
            output = torch.cat((output, output_dx, output_dxx, output_dx2), 1) # 1, 4, 36, H, W

            output = output.permute(0,2,1,3,4).contiguous() # 1, 36, 4, H, W

            output = self.conv1(output)
            output = self.conv2(output) + output

            output = output.view(output.size(0), -1, output.size(3), output.size(4))

            Handcrafted_Feature_Pyramid.append(output)


        # Build laterals
        laterals = [lateral_conv(Handcrafted_Feature_Pyramid[i])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='bilinear')

        # Build outputs
        out = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return out
