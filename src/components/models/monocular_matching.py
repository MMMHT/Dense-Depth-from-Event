import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from .refinement import StereoNetRefinement

from .feature_extractor import FeatureExtractor, Handcrafted_Feature
from .cost import CostVolume_Mono_Pyramid
from .aggregation import AdaptiveAggregation
from .estimation import DisparityEstimationPyramid


class EventmonocularNetwork(nn.Module):
    def __init__(self, in_channels):
        super(EventmonocularNetwork, self).__init__()
        self.in_channels = in_channels

        self.num_downsample = 1
        num_scales = 3
        num_fusions = 6
        deformable_groups = 1
        mdconv_dilation = 2
        no_intermediate_supervision = False
        num_stage_blocks = 1
        num_deform_blocks = 3
        no_mdconv = False

        self.feature_extractor = FeatureExtractor(in_channels=in_channels)
        self.cost_constructor = CostVolume_Mono_Pyramid(max_disp = in_channels)
        self.hand_crafted_cost_feature = Handcrafted_Feature(in_planes = in_channels)
        self.aggregation = AdaptiveAggregation(max_disp=in_channels,
                                               num_scales=num_scales,
                                               num_fusions=num_fusions,
                                               num_stage_blocks=num_stage_blocks,
                                               num_deform_blocks=num_deform_blocks,
                                               no_mdconv=no_mdconv,
                                               mdconv_dilation=mdconv_dilation,
                                               deformable_groups=deformable_groups,
                                               intermediate_supervision= not no_intermediate_supervision)

        self.disparity_estimation = DisparityEstimationPyramid(in_channels)

        # Refinement
        refine_module_list = nn.ModuleList()
        for i in range(self.num_downsample):
            refine_module_list.append(StereoNetRefinement(in_planes=self.in_channels))

        self.refinement = refine_module_list

    def disparity_refinement_monocular(self, event_input, disparity):
        disparity_pyramid = []
        warp_img_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1. / pow(2, self.num_downsample - i - 1)

            if scale_factor == 1.0:
                curr_img = event_input
            else:
                curr_img = F.interpolate(event_input,
                                              scale_factor=scale_factor,
                                              mode='bilinear', align_corners=False)

            disparity, warp_img = self.refinement[i](disparity, curr_img)
            disparity_pyramid.append(disparity)  # [H/2, H]
            warp_img_pyramid.append(warp_img)

        return disparity_pyramid, warp_img_pyramid

    def forward(self, event_image, groundtruth, params):

        essen = params['essen'][0]
        max_disp = params['max_disp'][0].item()

        faeture_pyramid = self.feature_extractor(event_image)
        cost_pyramid = self.cost_constructor(faeture_pyramid)
        cost_pyramid_hc = self.hand_crafted_cost_feature(cost_pyramid)
        adaptive_aggregated = self.aggregation(cost_pyramid_hc)
        disparity_pyramid = self.disparity_estimation(adaptive_aggregated)
        refined_disp_pyramid, warp_img_pyramid = self.disparity_refinement_monocular(event_image, disparity_pyramid[-1])

        disparity_pyramid += refined_disp_pyramid

         # generate the warpped input event image
        disp = disparity_pyramid[-1]
        disp_warp = disp.round()
        disp_warp[disp > max_disp] =max_disp
        disp_warp[disp < 0] = 0
        warp_img = torch.gather(event_image, 1, disp_warp.unsqueeze(0).long())
        warp_img = warp_img.permute(1,0,2,3)

        #  scale the disparity to the actual metric disparity
        for i in range(len(disparity_pyramid)):
            disparity_pyramid[i] = disparity_pyramid[i] * essen.view(-1, 1, 1).float()
            disparity_pyramid[i] = disparity_pyramid[i].unsqueeze(1)


        mask_disp = (groundtruth > 0)

        return disparity_pyramid[-1], mask_disp, warp_img



