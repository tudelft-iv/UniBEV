# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shiming Wang
# ---------------------------------------------
import copy
import os.path as osp

import mmcv.utils.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from mmdet3d.bevformer_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from mmdet3d.bevformer_plugin.models.utils.bricks import run_time
from .convolutional_channel_attention import ConvChannelAttention, MultiModalConvChannelAttention, ConvSpatialAttention
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionFusionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 use_align_loss=False,
                 use_feats_distil_loss=False,
                 fusion_method='linear',
                 drop_modality = None,
                 feature_norm = None,
                 spatial_norm = None,
                 with_modal_embedding=None,
                 finetune_init_value=None,
                 rotate_center=[100, 100],
                 bev_h = 200,
                 bev_w = 200,
                 vis_output = None,
                 **kwargs):
        super(PerceptionFusionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.vis_output = vis_output

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.use_align_loss=use_align_loss
        self.use_feats_distil_loss = use_feats_distil_loss
        self.finetune_init_value = finetune_init_value
        self.fusion_method = fusion_method
        self.with_modal_embedding = with_modal_embedding
        if self.use_feats_distil_loss:
            assert self.fusion_method == 'cat_pseudo_feats'
        if self.fusion_method == 'linear':
            self.scale_factor = 1
        elif self.fusion_method == 'avg':
            self.scale_factor = 1
        elif self.fusion_method == 'cat_pseudo_feats':
            self.scale_factor = 1
        elif self.fusion_method == 'cat':
            self.scale_factor = 2  # used to scale up the dimension when concatenate
        else:
            raise ValueError('Unrecognizable fusion method:{}'.format(self.fusion_method))
        self.drop_modality = drop_modality
        self.feature_norm = feature_norm
        self.spatial_norm = spatial_norm
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims * self.scale_factor, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.with_modal_embedding:
            self.modal_embbeding_mlp = nn.Sequential(
                nn.Linear(2, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True))
        if self.feature_norm is None:
            self.img_feature_norm = self.lidar_feature_norm = nn.Identity()
        elif self.feature_norm == 'SeparateBatchNorm':
            self.img_feature_norm = nn.BatchNorm2d(self.embed_dims)
            self.lidar_feature_norm = nn.BatchNorm2d(self.embed_dims)
        elif self.feature_norm == 'SharedBatchNorm':
            self.img_feature_norm = self.lidar_feature_norm = nn.BatchNorm2d(self.embed_dims)
        elif self.feature_norm == 'PointsLayerNorm':
            self.img_feature_norm = nn.Identity()
            self.lidar_feature_norm = nn.LayerNorm(self.embed_dims)
        elif self.feature_norm == 'ProjectionHeadLayerNorm':
            projection_head = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims))
            self.lidar_feature_norm = projection_head
            self.img_feature_norm = nn.Identity()
        elif self.feature_norm == 'ChannelAttention':
            self.lidar_feature_weights = nn.Parameter(torch.Tensor(self.embed_dims))
            self.img_feature_weights = nn.Parameter(torch.Tensor(self.embed_dims))
        elif self.feature_norm == 'SpatialAttention':
            self.lidar_feature_weights = nn.Parameter(torch.Tensor(self.bev_h, self.bev_w))
            self.img_feature_weights = nn.Parameter(torch.Tensor(self.bev_h * self.bev_w))

        elif self.feature_norm == 'ChannelAttentionNormalization':
            self.feature_norm_layer = nn.Softmax(dim=0)
            self.lidar_feature_weights = nn.Parameter(torch.Tensor(self.embed_dims))
            self.img_feature_weights = nn.Parameter(torch.Tensor(self.embed_dims))
        elif self.feature_norm == 'MLP_ChannelAttnNorm' or self.feature_norm == 'MLP_ChannelAttnNorm2':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 2),
                nn.ReLU(inplace=True))
            self.img_feature_norm = self.lidar_feature_norm = nn.Identity()
        elif self.feature_norm == 'MLP_ChannelAttnNorm3':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 2),
                nn.ReLU(inplace=True))
            self.img_feature_norm = self.lidar_feature_norm = nn.Identity()
        elif self.feature_norm == 'Conv_ChannelAttnNorm':
            self.softmax = nn.Softmax(dim=-1)
            self.img_feature_weights = ConvChannelAttention(self.embed_dims)
            self.lidar_feature_weights = ConvChannelAttention(self.embed_dims)
            self.img_feature_norm = self.lidar_feature_norm = nn.Identity()
        elif self.feature_norm == 'MM_Conv_ChannelAttnNorm':
            self.multi_modal_channel_attention = MultiModalConvChannelAttention(self.embed_dims)
            self.img_feature_norm = self.lidar_feature_norm = nn.Identity()
        else:
            raise ValueError('Unrecognized feature norm type:', self.feature_norm)

        if self.spatial_norm == 'ConvSpatialAttention':
            self.softmax=nn.Softmax(dim=1)
            self.img_spatial_weights = ConvSpatialAttention()
            self.lidar_spatial_weights = ConvSpatialAttention()
            self.img_feature_norm = self.lidar_feature_norm = nn.Identity()
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

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
        if self.feature_norm == 'ChannelAttention' or self.feature_norm == 'SpatialAttention':
            normal_(self.lidar_feature_weights)
            normal_(self.img_feature_weights)
        elif self.feature_norm == 'ChannelAttentionNormalization':
            if self.finetune_init_value is not None:
                nn.init.constant_(self.img_feature_weights, self.finetune_init_value)
                nn.init.constant_(self.lidar_feature_weights, self.finetune_init_value)
            else:
                normal_(self.lidar_feature_weights)
                normal_(self.img_feature_weights)
            # raise ValueError(self.lidar_feature_weights, self.finetune_init_value)
            # ## debug
            # print('In transformer init_weight layer:')
            # print(' lidar_feature_weights:', self.lidar_feature_weights)
            # print(list(self.parameters()))
            # raise ValueError('Checkpoint')
            # ##
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
        if self.with_modal_embedding:
            xavier_init(self.modal_embbeding_mlp, distribution='uniform', bias=0.)
        if self.feature_norm == 'MLP_ChannelAttnNorm' or self.feature_norm == 'MLP_ChannelAttnNorm2' or self.feature_norm == 'MLP_ChannelAttnNorm3':
            xavier_init(self.channel_weights_proj, distribution='uniform', bias=0.)
        elif self.feature_norm == 'Conv_ChannelAttnNorm':
            xavier_init(self.img_feature_weights)
            xavier_init(self.lidar_feature_weights)
        elif self.feature_norm == 'MM_Conv_ChannelAttnNorm':
            xavier_init(self.multi_modal_channel_attention)
        if self.spatial_norm == 'ConvSpatialAttention':
            xavier_init(self.img_spatial_weights)
            xavier_init(self.lidar_spatial_weights)

    def get_probability(self, prob):
        return True if np.random.random() < prob else False

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
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

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

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

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                pts_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        # ## debug
        # if self.training is False:
        #     self.lidar_feature_weights = torch.ones_like(self.lidar_feature_weights)
        #     self.img_feature_weights = torch.ones_like(self.img_feature_weights)
        # ##
        l_flag = 1
        c_flag = 1
        md_version = None
        # ##debug
        # print(self.drop_modality)
        # ##
        if self.drop_modality is not None:
            assert isinstance(self.drop_modality, dict)
            md_prob = self.drop_modality['prob']
            md_version = self.drop_modality['version']
            if self.training is True:
                v_flag = self.get_probability(md_prob)
                if v_flag:
                    l_flag = self.get_probability(md_prob)*1
                    c_flag = 1 - l_flag
            # ## debug
            # print('\nv_flag',v_flag)
            # print('\nc_flag',c_flag)
            # print('\nl_flag',l_flag)
            # ##
        if mlvl_feats is None:
            c_flag = 0
        if pts_feats is None:
            l_flag = 0
        if self.feature_norm == 'ChannelAttentionNormalization':
            feature_weight_list=[]
            feature_weight_list.append(self.img_feature_weights.unsqueeze(0))
            feature_weight_list.append(self.lidar_feature_weights.unsqueeze(0))
            feature_weights = torch.cat(feature_weight_list, 0)
            if c_flag == 1 and l_flag == 1:
                feature_weights_norm = self.feature_norm_layer(feature_weights)
                img_weights = feature_weights_norm[0]
                pts_weights = feature_weights_norm[1]
            else:
                img_weights = self.feature_norm_layer(feature_weights[0:1])[0]
                pts_weights = self.feature_norm_layer(feature_weights[1:2])[0]

        if mlvl_feats is not None:
            img_bev_embed = self.get_bev_features(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

            ori_img_bev_embed = img_bev_embed.clone()
            bs = mlvl_feats[0].size(0)
            if self.feature_norm == 'ChannelAttention':
                img_bev_embed = img_bev_embed.permute(1, 0, 2)
                if md_version == 'v2':
                    img_weights = torch.ones_like(self.img_feature_weights)[None, None, :] - l_flag * self.img_feature_weights[None, None, :].sigmoid()
                    img_bev_embed = img_bev_embed * img_weights
                else:
                    img_bev_embed = img_bev_embed * self.img_feature_weights[None, None, :].sigmoid()
            elif self.feature_norm == 'ChannelAttentionNormalization':
                img_bev_embed = img_bev_embed.permute(1, 0, 2)
                img_bev_embed = img_bev_embed * img_weights[None, None, :]
                # ##debug
                # print('\nIn image feats extractor:')
                # print('v_flag:', v_flag)
                # print('c_flag:', c_flag)
                # print('l_flag:', l_flag)
                # print('img_weights:', img_weights.size())
                # print('img_weights:', img_weights[0,0,0])
                # ##
            elif self.feature_norm == 'SpatialAttention':
                img_bev_embed = img_bev_embed.permute(1, 0, 2)
                if md_version == 'v2':
                    img_weights = torch.ones_like(self.img_feature_weights)[:, None, None] - l_flag * self.img_feature_weights[:, None, None].sigmoid()
                    img_bev_embed = img_bev_embed * img_weights
                else:
                    img_bev_embed = img_bev_embed * self.img_feature_weights[:, None, None].sigmoid()
            else:
                img_bev_embed = img_bev_embed.permute(0, 2, 1).view(bs, self.embed_dims, bev_h, bev_w)
                img_bev_embed = self.img_feature_norm(img_bev_embed)
                img_bev_embed = img_bev_embed.view(bs, self.embed_dims, bev_h*bev_w)
                img_bev_embed = img_bev_embed.permute(2, 0, 1) # (bev_h*bev_w, bs, embed_dims)
                #img_bev_embed = img_bev_embed.permute(1, 0, 2)
        else:
            img_bev_embed = None
            ori_img_bev_embed = None

        if pts_feats is not None:
            pts_feat = pts_feats[0] # (bs, embed_dims, bev_h, bev_w)

            ori_pts_feats = pts_feat.clone()
            bs, C_pts, H_pts, W_pts = pts_feat.size()
            if self.feature_norm == 'PointsLayerNorm' or self.feature_norm == 'ProjectionHeadLayerNorm':
                pts_feat = pts_feat.permute(0, 2, 3, 1) # (bs, bev_h, bev_w, embed_dims)
                # todo
                # view should be permute
                pts_feat = self.lidar_feature_norm(pts_feat)
                pts_feat = pts_feat.permute(0, 3, 1, 2) #(bs, C_pts, H_pts, W_pts)
                # todo
                # view should be permute
            elif self.feature_norm == 'ChannelAttention':
                if md_version == 'v2':
                    pts_weights = torch.ones_like(self.lidar_feature_weights)[None,:,None,None] - c_flag * self.lidar_feature_weights[None,:,None,None].sigmoid()
                    pts_feat = pts_feat * pts_weights
                else:
                    pts_feat = pts_feat * self.lidar_feature_weights[None,:,None,None].sigmoid()
                # ##debug
                # print('\nIn points feature extractor:')
                # print('v_flag:', v_flag)
                # print('c_flag:', c_flag)
                # print('l_flag:', l_flag)
                # print('pts_weights:', pts_weights.size())
                # print('pts_weights:', pts_weights[0,0,0,0])
                # ##
            elif self.feature_norm == 'SpatialAttention':
                if md_version == 'v2':
                    pts_weights = torch.ones_like(self.lidar_feature_weights)[None, None, :, :] - c_flag * self.lidar_feature_weights[None, None, :, :].sigmoid()
                    pts_feat = pts_feat * pts_weights
                else:
                    pts_feat = pts_feat * self.lidar_feature_weights[None, None, :, :].sigmoid()
            elif self.feature_norm == 'ChannelAttentionNormalization':
                    pts_feat = pts_feat * pts_weights[None, :, None, None]
            else:
                pts_feat = self.lidar_feature_norm(pts_feat)

            pts_feats_bev = pts_feat.permute(2, 3, 0, 1).view(H_pts*W_pts, bs, C_pts) # (bev_h*bev_w, bs, embed_dims)
        else:
            pts_feats_bev = None
            ori_pts_feats = None

        if self.feature_norm == 'MLP_ChannelAttnNorm':
            if l_flag == 0 and c_flag == 1:
                assert img_bev_embed is not None
                pts_feats_bev_ph = torch.zeros_like(img_bev_embed)
                input_bev_feats = torch.cat([img_bev_embed, pts_feats_bev_ph], dim=0).permute(1, 2, 0)
                multi_modal_weights = self.channel_weights_proj(input_bev_feats)
                channel_norm_attn = F.softmax(multi_modal_weights[:,:,0:1], dim=-1)
                img_feature_weight = channel_norm_attn.squeeze(-1)
                img_bev_embed = img_bev_embed * img_feature_weight
            elif l_flag == 1 and c_flag == 0:
                assert pts_feats_bev is not None
                img_bev_embed_ph = torch.zeros_like(pts_feats_bev)
                input_bev_feats = torch.cat([img_bev_embed_ph, pts_feats_bev], dim=0).permute(1, 2, 0)
                multi_modal_weights = self.channel_weights_proj(input_bev_feats)
                channel_norm_attn = F.softmax(multi_modal_weights[:, :, 1:2], dim=-1)
                pts_feature_weight = channel_norm_attn.squeeze(-1)
                pts_feats_bev = pts_feats_bev * pts_feature_weight
            elif l_flag == 1 and c_flag == 1:
                assert img_bev_embed is not None and pts_feats_bev is not None
                input_bev_feats = torch.cat([img_bev_embed, pts_feats_bev], dim=0).permute(1, 2, 0)
                multi_modal_weights = self.channel_weights_proj(input_bev_feats)
                channel_norm_attn = F.softmax(multi_modal_weights, dim=-1)
                img_feature_weight = channel_norm_attn[:, :, 0]
                pts_feature_weight = channel_norm_attn[:, :, 1]
                img_bev_embed = img_bev_embed * img_feature_weight
                pts_feats_bev = pts_feats_bev * pts_feature_weight
            else:
                raise ValueError('Unrecognized flag values: c_flag {}, l_falg {}.'.format(c_flag, l_flag))
        elif self.feature_norm == 'MLP_ChannelAttnNorm2':
            if l_flag == 0 and c_flag == 1:
                assert img_bev_embed is not None
                img_bev_embed = img_bev_embed * c_flag
                if pts_feats_bev is not None:
                    pts_feats_bev = pts_feats_bev * l_flag
                else:
                    pts_feats_bev = torch.zeros_like(img_bev_embed)
                input_bev_feats = torch.cat([img_bev_embed, pts_feats_bev], dim=0).permute(1, 2, 0)
                multi_modal_weights = self.channel_weights_proj(input_bev_feats)
                channel_norm_attn = F.softmax(multi_modal_weights[:,:,0:1], dim=-1)
                img_feature_weight = channel_norm_attn.squeeze(-1)
                img_bev_embed = img_bev_embed * img_feature_weight
            elif l_flag == 1 and c_flag == 0:
                assert pts_feats_bev is not None
                pts_feats_bev = pts_feats_bev * l_flag
                if img_bev_embed is not None:
                    img_bev_embed = img_bev_embed * c_flag
                else:
                    img_bev_embed = torch.zeros_like(pts_feats_bev)
                input_bev_feats = torch.cat([img_bev_embed, pts_feats_bev], dim=0).permute(1, 2, 0)
                multi_modal_weights = self.channel_weights_proj(input_bev_feats)
                channel_norm_attn = F.softmax(multi_modal_weights[:, :, 1:2], dim=-1)
                pts_feature_weight = channel_norm_attn.squeeze(-1)
                pts_feats_bev = pts_feats_bev * pts_feature_weight
            elif l_flag == 1 and c_flag == 1:
                assert img_bev_embed is not None and pts_feats_bev is not None
                input_bev_feats = torch.cat([img_bev_embed, pts_feats_bev], dim=0).permute(1, 2, 0)
                multi_modal_weights = self.channel_weights_proj(input_bev_feats)
                channel_norm_attn = F.softmax(multi_modal_weights, dim=-1)
                img_feature_weight = channel_norm_attn[:, :, 0]
                pts_feature_weight = channel_norm_attn[:, :, 1]
                img_bev_embed = img_bev_embed * img_feature_weight
                pts_feats_bev = pts_feats_bev * pts_feature_weight
            else:
                raise ValueError('Unrecognized flag values: c_flag {}, l_falg {}.'.format(c_flag, l_flag))
        elif self.feature_norm == 'MLP_ChannelAttnNorm3':
            if l_flag == 0 and c_flag == 1:
                assert img_bev_embed is not None
                img_bev_embed = img_bev_embed * c_flag
                if pts_feats_bev is not None:
                    pts_feats_bev = pts_feats_bev * l_flag
                else:
                    pts_feats_bev = torch.zeros_like(img_bev_embed)
                input_bev_feats = torch.cat([img_bev_embed, pts_feats_bev], dim=0).permute(1, 2, 0)
                multi_modal_weights = self.channel_weights_proj(input_bev_feats)
                channel_norm_attn = F.softmax(multi_modal_weights[:,:,0:1], dim=-1)
                img_feature_weight = channel_norm_attn.squeeze(-1)
                img_bev_embed = img_bev_embed * img_feature_weight
            elif l_flag == 1 and c_flag == 0:
                assert pts_feats_bev is not None
                pts_feats_bev = pts_feats_bev * l_flag
                if img_bev_embed is not None:
                    img_bev_embed = img_bev_embed * c_flag
                else:
                    img_bev_embed = torch.zeros_like(pts_feats_bev)
                input_bev_feats = torch.cat([img_bev_embed, pts_feats_bev], dim=0).permute(1, 2, 0)
                multi_modal_weights = self.channel_weights_proj(input_bev_feats)
                channel_norm_attn = F.softmax(multi_modal_weights[:, :, 1:2], dim=-1)
                pts_feature_weight = channel_norm_attn.squeeze(-1)
                pts_feats_bev = pts_feats_bev * pts_feature_weight
            elif l_flag == 1 and c_flag == 1:
                assert img_bev_embed is not None and pts_feats_bev is not None
                input_bev_feats = torch.cat([img_bev_embed, pts_feats_bev], dim=0).permute(1, 2, 0)
                multi_modal_weights = self.channel_weights_proj(input_bev_feats)
                channel_norm_attn = F.softmax(multi_modal_weights, dim=-1)
                img_feature_weight = channel_norm_attn[:, :, 0]
                pts_feature_weight = channel_norm_attn[:, :, 1]
                img_bev_embed = img_bev_embed * img_feature_weight
                pts_feats_bev = pts_feats_bev * pts_feature_weight
            else:
                raise ValueError('Unrecognized flag values: c_flag {}, l_falg {}.'.format(c_flag, l_flag))
        elif self.feature_norm == 'Conv_ChannelAttnNorm':
            if l_flag == 0 and c_flag == 1:
                assert img_bev_embed is not None
                img_bev_embed = img_bev_embed * c_flag
                if pts_feats_bev is not None:
                    pts_feats_bev = pts_feats_bev * l_flag
                else:
                    pts_feats_bev = torch.zeros_like(img_bev_embed)

                img_feature_weight = self.img_feature_weights(img_bev_embed.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).squeeze(-1)
                pts_feature_weight = self.lidar_feature_weights(pts_feats_bev.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).squeeze(-1).squeeze(-1)

                img_norm_weight = self.softmax(img_feature_weight).squeeze(-1)

                img_bev_embed = img_bev_embed * img_norm_weight
                pts_feats_bev = pts_feats_bev * pts_feature_weight
            elif l_flag == 1 and c_flag == 0:
                assert pts_feats_bev is not None
                pts_feats_bev = pts_feats_bev * l_flag
                if img_bev_embed is not None:
                    img_bev_embed = img_bev_embed * c_flag
                else:
                    img_bev_embed = torch.zeros_like(pts_feats_bev)
                img_feature_weight = self.img_feature_weights(img_bev_embed.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).squeeze(-1).squeeze(-1)
                pts_feature_weight = self.lidar_feature_weights(pts_feats_bev.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).squeeze(-1)

                pts_norm_weight = self.softmax(pts_feature_weight).squeeze(-1)

                img_bev_embed = img_bev_embed * img_feature_weight
                pts_feats_bev = pts_feats_bev * pts_norm_weight

            elif l_flag == 1 and c_flag == 1:
                assert img_bev_embed is not None and pts_feats_bev is not None
                img_feature_weight = self.img_feature_weights(img_bev_embed.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).squeeze(-1)
                pts_feature_weight = self.lidar_feature_weights(pts_feats_bev.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).squeeze(-1)

                multi_modal_weights = torch.cat([img_feature_weight, pts_feature_weight], dim=-1)
                multi_modal_norm_weights = self.softmax(multi_modal_weights)

                img_norm_weight = multi_modal_norm_weights[:,:,0]
                pts_norm_weight = multi_modal_norm_weights[:,:,1]

                img_bev_embed = img_bev_embed * img_norm_weight
                pts_feats_bev = pts_feats_bev * pts_norm_weight
        elif self.feature_norm == 'MM_Conv_ChannelAttnNorm':
            if l_flag == 0 and c_flag == 1:
                assert img_bev_embed is not None
                mm_feature_weights = self.multi_modal_channel_attention(img_bev_embed.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h), None)
                img_bev_embed = img_bev_embed * mm_feature_weights.squeeze(-1)

            elif l_flag == 1 and c_flag == 0:
                assert pts_feats_bev is not None
                mm_feature_weights = self.multi_modal_channel_attention(None, pts_feats_bev.permute(1, 2, 0).view(bs, self.embed_dims, self.bev_w, self.bev_h))
                pts_feats_bev = pts_feats_bev * mm_feature_weights.squeeze(-1)

            elif l_flag == 1 and c_flag == 1:
                assert img_bev_embed is not None and pts_feats_bev is not None
                mm_feature_weights = self.multi_modal_channel_attention(img_bev_embed.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h),
                                                                        pts_feats_bev.permute(1, 2, 0).view(bs, self.embed_dims, self.bev_w, self.bev_h))

                img_bev_embed = img_bev_embed * mm_feature_weights[...,0]
                pts_feats_bev = pts_feats_bev * mm_feature_weights[...,1]

        if self.spatial_norm == 'ConvSpatialAttention':
            if l_flag == 0 and c_flag == 1:
                assert img_bev_embed is not None
                img_bev_embed = img_bev_embed * c_flag
                if pts_feats_bev is not None:
                    pts_feats_bev = pts_feats_bev * l_flag
                else:
                    pts_feats_bev = torch.zeros_like(img_bev_embed)

                img_spatial_weight = self.img_spatial_weights(img_bev_embed.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).flatten(2)
                pts_spatial_weight = self.lidar_spatial_weights(pts_feats_bev.permute(1,2,0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).flatten(2).permute(2,0,1)

                img_spatial_norm_weight = self.softmax(img_spatial_weight).permute(2,0,1)

                img_bev_embed = img_bev_embed * img_spatial_norm_weight
                pts_feats_bev = pts_feats_bev * pts_spatial_weight

            elif l_flag == 1 and c_flag == 0:
                assert pts_feats_bev is not None
                pts_feats_bev = pts_feats_bev * l_flag
                if img_bev_embed is not None:
                    img_bev_embed = img_bev_embed * c_flag
                else:
                    img_bev_embed = torch.zeros_like(pts_feats_bev)

                img_spatial_weight = self.img_spatial_weights(img_bev_embed.permute(1, 2, 0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).flatten(2).permute(2, 0, 1)
                pts_spatial_weight = self.lidar_spatial_weights(pts_feats_bev.permute(1, 2, 0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).flatten(2)

                pts_spatial_norm_weight = self.softmax(pts_spatial_weight).permute(2,0,1)

                img_bev_embed = img_bev_embed * img_spatial_weight
                pts_feats_bev = pts_feats_bev * pts_spatial_norm_weight
            elif l_flag == 1 and c_flag == 1:
                assert img_bev_embed is not None and pts_feats_bev is not None
                img_spatial_weight = self.img_spatial_weights(img_bev_embed.permute(1, 2, 0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).flatten(2)
                pts_spatial_weight = self.lidar_spatial_weights(pts_feats_bev.permute(1, 2, 0).view(bs, self.embed_dims, self.bev_w, self.bev_h)).flatten(2)

                multi_modal_spatial_weights = torch.cat([img_spatial_weight, pts_spatial_weight], dim=1).permute(2,0,1)
                multi_modal_norm_weights = self.softmax(multi_modal_spatial_weights)

                img_spatial_norm_weight = multi_modal_norm_weights[:, :, 0:1]
                pts_spatial_norm_weight = multi_modal_norm_weights[:, :, 1:]

                img_bev_embed = img_bev_embed * img_spatial_norm_weight
                pts_feats_bev = pts_feats_bev * pts_spatial_norm_weight

        query_pos, query = torch.split(object_query_embed, self.embed_dims * self.scale_factor, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        img_metas = kwargs['img_metas'][0]

        if pts_feats_bev is not None and img_bev_embed is not None:
            if self.fusion_method == 'linear':
                assert pts_feats_bev.size() == img_bev_embed.size()
                bev_embed = img_bev_embed * c_flag + pts_feats_bev * l_flag
            elif self.fusion_method == 'avg':
                assert pts_feats_bev.size() == img_bev_embed.size()
                bev_embed = img_bev_embed * c_flag / (c_flag+l_flag) + pts_feats_bev * l_flag / (c_flag+l_flag)
            elif self.fusion_method == 'cat':
                bev_embed = torch.cat((img_bev_embed*c_flag, pts_feats_bev*l_flag), -1)
                # ## debug
                # print('\nbev_embed:', bev_embed)
                # print('\nbev_embed:', bev_embed)
                # ##
            elif self.fusion_method == 'cat_pseudo_feats':
                c_pseudo_flag = 1 - c_flag
                l_pseudo_flag = 1 - l_flag

                c_true_weights = torch.Tensor([c_flag]).expand(self.embed_dims//2)
                l_psudo_weights = torch.Tensor([l_pseudo_flag]).expand(self.embed_dims//2)
                l_true_weights = torch.Tensor([l_flag]).expand(self.embed_dims//2)
                c_pseudo_weights = torch.Tensor([c_pseudo_flag]).expand(self.embed_dims//2)

                img_flags = torch.cat((c_true_weights, l_psudo_weights)).cuda()
                pts_flags = torch.cat((c_pseudo_weights, l_true_weights)).cuda()

                bev_embed = img_bev_embed*img_flags + pts_feats_bev*pts_flags

                # if c_flag == 1 and l_flag == 1:
                #     assert bev_embed.equal(torch.cat((img_bev_embed[..., :self.embed_dims//2], pts_feats_bev[..., self.embed_dims//2:]), -1))
                # elif c_flag == 1 and l_flag == 0:
                #     assert bev_embed.equal(img_bev_embed)
                # elif c_flag ==0 and l_flag == 1:
                #     assert bev_embed.equal(pts_feats_bev)
                # else:
                #     raise ValueError('Unsupported flag values: {}'.format(c_flag, l_flag))
            else:
                raise ValueError('Unrecognizable fusion method:{}'.format(self.fusion_method))
        elif img_bev_embed is not None:
            if self.fusion_method == 'cat':
                pts_feats_bev = torch.zeros_like(img_bev_embed)
                bev_embed = torch.cat((img_bev_embed, pts_feats_bev), -1)
            else:
                bev_embed = img_bev_embed
        elif pts_feats_bev is not None:
            if self.fusion_method == 'cat':
                img_bev_embed = torch.zeros_like(pts_feats_bev)
                bev_embed = torch.cat((img_bev_embed, pts_feats_bev), -1)
            else:
                bev_embed = pts_feats_bev
        else:
            raise ValueError('At least one type of bev features is needed.')

        if self.with_modal_embedding:
            modal_status=torch.Tensor([c_flag, l_flag]).cuda()

            modal_embedding = self.modal_embbeding_mlp(modal_status)

            bev_embed += modal_embedding
        ##Visualization Part in Test
        if self.training is False and self.vis_output is not None:
            assert isinstance(self.vis_output, dict)
            outdir = self.vis_output['outdir']
            pts_path = img_metas['pts_filename']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            result_path = osp.join(outdir, file_name)
            mmcv.utils.path.mkdir_or_exist(result_path)

            vis_data = {}
            for key in self.vis_output['keys']+self.vis_output['special_keys']:
                vis_data[key] = locals()[key]
            for attr in self.vis_output['attrs']:
                vis_data[attr] = getattr(self, attr)

            vis_data.update(dict(lidar_file_name=file_name))
            torch.save(vis_data, osp.join(result_path, 'vis_data.pt'))
        ##
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)


        inter_references_out = inter_references
        if self.use_align_loss or self.use_feats_distil_loss:
            return ori_pts_feats, ori_img_bev_embed, bev_embed, inter_states, init_reference_out, inter_references_out
        else:
            return bev_embed, inter_states, init_reference_out, inter_references_out
