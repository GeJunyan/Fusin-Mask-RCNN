# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import copy

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        #self.backbone = build_backbone(cfg)
        self.backbone_rgb = build_backbone(cfg)
        self.backbone_hha = build_backbone(cfg)

        #把维数降下来
        self.down1 = nn.ConvTranspose2d(256*2, 256, 1, 1)
        self.down2 = nn.ConvTranspose2d(256*2, 256, 1, 1)
        self.down3 = nn.ConvTranspose2d(256*2, 256, 1, 1)
        self.down4 = nn.ConvTranspose2d(256*2, 256, 1, 1)
        self.down5 = nn.ConvTranspose2d(256*2, 256, 1, 1)

        #self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.rpn = build_rpn(cfg, 256)
       
        #self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, 256)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")


        images_hha = copy.deepcopy(images)#
        images_rgb = copy.deepcopy(images)#
        images_hha.tensors = images_hha.tensors[:,0:3,:,:]
        images_rgb.tensors = images_rgb.tensors[:,0:3,:,:]
        
        
        images_hha = to_image_list(images_hha)
        images_rgb = to_image_list(images_rgb)


        features_hha = self.backbone_hha(images_hha.tensors)
        features_rgb = self.backbone_rgb(images_rgb.tensors)
        
        #将五个layer的通道分别融合 然后降维
        features_hha_rgb = ()
        features_hha_rgb = list(features_hha_rgb)
        features_hha_rgb.append(torch.cat((features_hha[0], features_rgb[0]), axis=1))
        features_hha_rgb.append(torch.cat((features_hha[1], features_rgb[1]), axis=1))
        features_hha_rgb.append(torch.cat((features_hha[2], features_rgb[2]), axis=1))
        features_hha_rgb.append(torch.cat((features_hha[3], features_rgb[3]), axis=1))
        features_hha_rgb.append(torch.cat((features_hha[4], features_rgb[4]), axis=1))
        
        features_hha_rgb[0] = self.down1(features_hha_rgb[0])
        features_hha_rgb[1] = self.down2(features_hha_rgb[1])
        features_hha_rgb[2] = self.down3(features_hha_rgb[2])
        features_hha_rgb[3] = self.down4(features_hha_rgb[3])
        features_hha_rgb[4] = self.down5(features_hha_rgb[4])


        proposals, proposal_losses = self.rpn(images, features_hha_rgb, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features_hha_rgb, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features_hha_rgb
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
