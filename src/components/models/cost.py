import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))




class CostVolume_Mono_Pyramid(nn.Module):
    def __init__(self, max_disp, window_size=7):
        super(CostVolume_Mono_Pyramid, self).__init__()
        self.ws = window_size
        self.max_disp = max_disp

        self.padding = window_size // 2

        self.max_disp = max_disp

        self.expansion = 11

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.max_disp * self.expansion, self.max_disp * self.expansion, kernel_size=3, stride=1, padding=1, groups=self.max_disp, bias=False),
            nn.GroupNorm(self.max_disp, self.max_disp* self.expansion),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.max_disp * self.expansion, self.max_disp * self.expansion, kernel_size=3, stride=1,padding=1, groups=self.max_disp, bias=False),
            nn.GroupNorm(self.max_disp, self.max_disp * self.expansion),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.max_disp* self.expansion, self.max_disp * self.expansion, kernel_size=3, stride=1, padding=1, groups=self.max_disp, bias=False),
            nn.GroupNorm(self.max_disp, self.max_disp * self.expansion),
            nn.LeakyReLU(inplace=True)
        )


        self.conv4 = nn.Sequential(
            nn.Conv2d(self.max_disp * self.expansion, self.max_disp, kernel_size=1, stride=1, groups=self.max_disp, bias=False),
            nn.BatchNorm2d(self.max_disp),
            nn.LeakyReLU(inplace=True)
        )

        self.kernel_x = nn.Parameter(torch.zeros(self.max_disp, 1, 3, 3))

        self.kernel_x.data[:, :,  0, 1] = -1
        self.kernel_x.data[:, :,  1, 1] = 2
        self.kernel_x.data[:, :,  2, 1] = -1

        self.kernel_y = nn.Parameter(torch.zeros(self.max_disp, 1, 3, 3))

        self.kernel_y.data[:, :, 1, 0] = -1
        self.kernel_y.data[:, :, 1, 1] = 2
        self.kernel_y.data[:, :, 1, 2] = -1

        self.kernel_x.requires_grad = False
        self.kernel_y.requires_grad = False


    def forward(self, x):

        num_scale = len(x)
        cost_volume_pyramid = []
        for s in range(num_scale):

            feature_input = x[s]

            c, d, h, w = feature_input.size()

            blur_dx = nn.functional.conv2d(feature_input, self.kernel_x, padding=1, groups=self.max_disp)
            blur_dy = nn.functional.conv2d(feature_input, self.kernel_y, padding=1, groups=self.max_disp)

            blur_dxx = nn.functional.conv2d(blur_dx, self.kernel_x, padding=1, groups=self.max_disp)
            blur_dyy = nn.functional.conv2d(blur_dy, self.kernel_y, padding=1, groups=self.max_disp)

            blur_dxy = nn.functional.conv2d(blur_dx, self.kernel_y, padding=1, groups=self.max_disp)

            blur_dx = blur_dx.unsqueeze(1)
            blur_dy = blur_dy.unsqueeze(1)
            blur_dxx = blur_dxx.unsqueeze(1)
            blur_dyy = blur_dyy.unsqueeze(1)
            blur_dxy = blur_dxy.unsqueeze(1)


            blur_dx2 = blur_dx ** 2
            blur_dy2 = blur_dy ** 2
            blur_dxx2 = blur_dxx ** 2
            blur_dyy2 = blur_dyy ** 2
            blur_dxy2 = blur_dxy ** 2

            blur_dxxyy = blur_dxx * blur_dyy

            blury_feature = torch.cat((blur_dx, blur_dy, blur_dxx, blur_dyy, blur_dxy
                                       ,blur_dx2, blur_dy2, blur_dxx2, blur_dyy2, blur_dxy2, blur_dxxyy), dim = 1)


            blury_feature = blury_feature.reshape(c, -1, h, w).contiguous()

            out = self.conv1(blury_feature)
            out = self.conv2(out)
            out = self.conv3(out)

            out = self.conv4(out)

            cost_volume_pyramid.append(out)

        return cost_volume_pyramid

