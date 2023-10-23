# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shiming Wang
# ---------------------------------------------
import time
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate

from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, \
    build_positional_encoding
from mmcv.cnn import ConvModule, xavier_init
from mmdet.models import DETECTORS

import mmdet3d
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from .bevformer import BEVFormer
from ..modules.temporal_self_attention import TemporalSelfAttention
from ..modules.spatial_cross_attention  import MSDeformableAttention3D
from ..modules.decoder import CustomMSDeformableAttention
from mmdet3d.bevformer_plugin.models.utils.grid_mask import GridMask
from mmdet3d.bevformer_plugin.models.utils.bricks import run_time

class SE_Block(nn.Module):
    def __init__(self, c):
        super(SE_Block, self).__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

@DETECTORS.register_module()
class BEVFormerCenterPointFusion(MVXTwoStageDetector):
    """
    BEVFormer_Fusion model: a multi-modal fusion model based on BEVFormer (cam) and PointPillars (lidar).
    BEVFormer_CenterHead model: a monocular 3D object detection Head using BEVFormer as the image features and BEV features encoder and the CenterHead as the decoder
    Args:
        pass: placeholder here
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 # BEVFusion params
                 imc=256,
                 lic=256,

                 # BEVFormer params
                 num_feature_levels = 4,
                 encoder = None,
                 num_cams =6,
                 bev_h = 200,
                 bev_w = 200,
                 embed_dims = 256,
                 use_shift = True,
                 use_can_bus = True,
                 can_bus_norm = True,
                 rotate_prev_bev = True,
                 rotate_center = [100, 100],
                 use_cams_embeds = True,
                 positional_encoding = None
                 ):
        super(BEVFormerCenterPointFusion, self).__init__(pts_voxel_layer,
                                                         pts_voxel_encoder,
                                                         pts_middle_encoder,
                                                         pts_fusion_layer,
                                                         img_backbone,
                                                         pts_backbone,
                                                         img_neck,
                                                         pts_neck,
                                                         pts_bbox_head,
                                                         img_roi_head,
                                                         img_rpn_head,
                                                         train_cfg,
                                                         test_cfg,
                                                         pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        ### properties from BEVFormer, Shiming Wang
        self.encoder = build_transformer_layer_sequence(encoder)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.rotate_prev_bev = rotate_prev_bev
        self.rotate_center = rotate_center
        self.use_cams_embeds = use_cams_embeds
        self.pc_range = self.encoder.pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_feature_levels = num_feature_levels
        ###
        self.fp16_enabled = False
        # temporal information
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.imc = imc
        self.lic = lic

        if self.with_pts_backbone and self.with_img_backbone:
            self.fusion_mode = True
        else:
            self.fusion_mode = False
        self.init_layers()

    def init_layers(self):
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
        if self.fusion_mode:
            self.reduc_conv = ConvModule(
                self.lic + self.imc,
                self.lic,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
            self.seblock = SE_Block(self.lic)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)


    def extract_img_feat(self, img, img_metas=None, len_queue=None):
        """Extract features of images."""
        if not self.with_img_backbone:
            return None
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def get_img_bev_features(self,
                             mlvl_feats,
                             bev_queries,
                             bev_h,
                             bev_w,
                             grid_length=[0.512, 0.512],
                             bev_pos=None,
                             prev_bev=None,
                             **kwargs):
        """
        get image bev features from BEVFormer method
        """
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        ## new_properties_shiming
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                            for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                            for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
                  np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
                  np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                          None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        ## new_properties_shiming
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_queries.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, img, points, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return img_feats, pts_feats

    @torch.no_grad()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_dummy(self, img, points):
        dummy_metas = None
        return self.forward_test(img=img, points=points, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list, bev_queries, bev_pos):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.get_img_bev_features(
                    mlvl_feats = img_feats,
                    bev_queries = bev_queries,
                    bev_h = self.bev_h,
                    bev_w = self.bev_w,
                    bev_pos = bev_pos,
                    prev_bev=prev_bev,
                    img_metas=img_metas)

            self.train()
            return prev_bev

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if img is not None:
            len_queue = img.size(1)
            prev_img = img[:, :-1, ...]
            img = img[:, -1, ...]

        img_feats, pts_feats = self.extract_feat(img=img, points=points, img_metas=img_metas)

        if img_feats is not None:
            bs,_, _ ,_, _= img_feats[0].size()
            dtype = img_feats[0].dtype
            bev_queries = self.bev_embedding.weight.to(dtype)
            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)

            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img,
                                               prev_img_metas,
                                               bev_queries,
                                               bev_pos)

            img_metas = [each[len_queue - 1] for each in img_metas]
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None

            img_bev_feats = self.get_img_bev_features(img_feats,
                                                      bev_queries,
                                                      self.bev_h,
                                                      self.bev_w,
                                                      grid_length=[self.real_h / self.bev_h, self.real_w / self.bev_w],
                                                      bev_pos=bev_pos,
                                                      img_metas=img_metas,
                                                      prev_bev=prev_bev)

            img_bev_feats = img_bev_feats.permute(0,2,1).view(bs,
                                                              self.embed_dims,
                                                              self.bev_h,
                                                              self.bev_w)
        # ## debug in BEVFormer_CenterHead
        # print('\nIn BEVFormer_CenterPoint:')
        # print(' img_bev_feats in detector:', img_bev_feats.size())
        # print(' pts_feats length in detector:', len(pts_feats))
        # print(' pts_feats in detector:', pts_feats[0].size())
        # ##
        if self.fusion_mode:
            if img_bev_feats.shape[2:] != pts_feats[0].shape[2:]:
                img_bev_feats = F.interpolate(img_bev_feats, pts_feats[0].shape[2:], mode='bilinear', align_corners=True)

        if self.fusion_mode:
            bev_feats = [self.reduc_conv(torch.cat([img_bev_feats, pts_feats[0]], dim=1))]
            bev_feats = [self.seblock(bev_feats[0])]
        elif pts_feats is not None:
            bev_feats = pts_feats
        elif img_bev_feats is not None:
            bev_feats = [img_bev_feats]
        else:
            raise ValueError('At least one modality is needed!')

        # ## debug
        # print('In BEVFormer_CenterHead:')
        # print(' points:', pts_feats)
        # print(' images:', img_bev_feats.size())
        # print(' bev features:', bev_feats[0].size())
        # print( 'bev features len:', len(bev_feats))
        # print( 'bev feature shape:', bev_feats[0].size())
        # ##
        losses = dict()
        losses_pts = self.forward_pts_train(bev_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        return losses

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        if points is not None:
            for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))
            num_augs = len(points)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(points), len(img_metas)))
        else:
            for var, name in [(img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))


        img = [img] if img is None else img
        pts = [points] if points is None else points

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            pts[0], img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test(self, points, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(img=img, points=points, img_metas=img_metas)
        bs,_, _ ,_, _= img_feats[0].size()
        dtype = img_feats[0].dtype

        bbox_list = [dict() for i in range(len(img_metas))]

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        if img_feats is not None:
            img_bev_feats = self.get_img_bev_features(img_feats,
                                                      bev_queries,
                                                      self.bev_h,
                                                      self.bev_w,
                                                      grid_length=[self.real_h / self.bev_h, self.real_w / self.bev_w],
                                                      bev_pos=bev_pos,
                                                      img_metas=img_metas,
                                                      prev_bev=prev_bev)
            new_prev_bev = img_bev_feats
            img_bev_feats = img_bev_feats.permute(0,2,1).view(bs,
                                                              self.embed_dims,
                                                              self.bev_h,
                                                              self.bev_w)
        else:
            img_bev_feats = None
            new_prev_bev = None

        if self.fusion_mode:
            bev_feats = [self.reduc_conv(torch.cat([img_bev_feats, pts_feats[0]], dim=1))]
            bev_feats = [self.seblock(pts_feats[0])]
        elif img_bev_feats is not None:
            bev_feats = [img_bev_feats]
        elif pts_feats is not None:
            bev_feats = pts_feats

        outs = self.pts_bbox_head(bev_feats)

        bbox_list_head = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_pts = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list_head
        ]

        # print('New pre bev features size:', new_prev_bev.size())
        ### hard-coded here: for evaluation
        # if self.fusion_method == 'cat':
        #     dim=new_prev_bev.size(-1)
        #     cam_prev_bev = new_prev_bev[:,:,:dim//2]
        #     new_prev_bev = cam_prev_bev

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
