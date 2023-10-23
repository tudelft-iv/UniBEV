# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
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
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
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
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
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
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

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
        shift = bev_queries.new_tensor(np.array([shift_x, shift_y])).permute(1, 0)  # xy, bs -> bs, xy
        # ## debug
        # print('\n'
        #       'In transformer: \n'
        #       'len mlvl_features: {}, \n'
        #       'mlvl features: {}, \n'
        #       'bev queries: {}, \n'
        #       'bev pos: {}, \n'
        #       'delta_x: {}, \n'
        #       'delta_y: {}, \n'
        #       'ego_angle: {}, \n'
        #       'grid_length: y {}, x {}, \n'
        #       'translation length: {}, \n'
        #       'translation angle: {}, \n'
        #       'shift_x: {}, \n'
        #       'shift y: {}, \n'
        #       'shift: {}. \n'
        #       ''.format(len(mlvl_feats),
        #                 mlvl_feats[0].size(),
        #                 bev_queries.size(),
        #                 bev_pos.size(),
        #                 np.around(delta_x, decimals=8),
        #                 np.around(delta_y,decimals=8),
        #                 ego_angle,
        #                 grid_length_y,
        #                 grid_length_x,
        #                 translation_length,
        #                 translation_angle,
        #                 shift_x,
        #                 shift_y,
        #                 shift.size()))
        # ##
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
        # ## debug
        # can_bus_original = copy.deepcopy(can_bus)
        # ##
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
        # ## debug
        # feat_flatten_list = copy.deepcopy(feat_flatten)
        # ##

        feat_flatten = torch.cat(feat_flatten, 2)

        # ## debug
        # feat_flatten_original = copy.deepcopy(feat_flatten)
        # ##

        ## new_properties_shiming
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_queries.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # ## debug
        # print('\n'
        #       'In transformer, variables before encoder: \n'
        #       'original can bus info: {}, \n'
        #       'can bus info after mlp: {}, \n'
        #       'feat_flatten list: {}, \n'
        #       'feat_flatten size: {}, \n'
        #       'cat feat flatten size: {}, \n'
        #       'cat feat flatten permute size: {}, \n'
        #       'spatial size: {}, \n'
        #       'level start index: {},\n'
        #       'camera embeddings: {}, \n'
        #       'level embeddings: {}, \n'
        #       ''.format(can_bus_original.size(),
        #                 can_bus.size(),
        #                 len(feat_flatten_list),
        #                 feat_flatten_list[0].size(),
        #                 feat_flatten_original.size(),
        #                 feat_flatten.size(),
        #                 spatial_shapes.size(),
        #                 level_start_index.size(),
        #                 self.cams_embeds.size(),
        #                 self.level_embeds.size()))
        # ##

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

        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        # ##Visualization Part in Test
        # if self.training is False:
        #     img_metas = kwargs['img_metas'][0]
        #     out_dir = './outputs/inference/bevformer_small_nus_C_1_4_val_mini_visualization' # hardcoded
        #     pts_path = img_metas['pts_filename']
        #     file_name = osp.split(pts_path)[-1].split('.')[0]
        #     result_path = osp.join(out_dir, file_name)
        #
        #     print('\nMode:', self.training)
        #     print('Query Size:', query.size())
        #     print('Query Pose Size:', query_pos.size())
        #     print('Reference Point:', reference_points.size())
        #     print('Image BEV features:', bev_embed.size())
        #     print('Image metas:', img_metas.keys())
        #     # print('Image metas filename:', img_metas['filename'])
        #     print('Image metas lidar2img:', len(img_metas['lidar2img']), img_metas['lidar2img'][0].shape)
        #     print('Image metas pts_filename:', img_metas['pts_filename'])
        #     print('Save dir:', result_path)
        #
        #     vis_data = dict(
        #         lidar_file_name=file_name,
        #         query = query,
        #         query_pos = query_pos,
        #         reference_points = reference_points,
        #         img_bev_feats = bev_embed)
        #     torch.save(vis_data, osp.join(result_path, 'vis_data.pt'))
        # ##

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

        # ##Visualization Part in Test
        # if self.training is False:
        #     img_metas = kwargs['img_metas'][0]
        #     out_dir = './outputs/inference/bevformer_small_nus_C_1_4_val_mini_visualization' # hardcoded
        #     pts_path = img_metas['pts_filename']
        #     file_name = osp.split(pts_path)[-1].split('.')[0]
        #     result_path = osp.join(out_dir, file_name)
        #
        #     print('\nMode:', self.training)
        #     print('Query Size:', query.size())
        #     print('Query Pose Size:', query_pos.size())
        #     print('Reference Point:', reference_points.size())
        #     print('Image BEV features:', bev_embed.size())
        #     print('Image metas:', img_metas.keys())
        #     print('Output of decoder:')
        #     print('Inter states:', inter_states.size())
        #     print('Inter reference:', inter_references.size())
        #     # print('Image metas filename:', img_metas['filename'])
        #     # print('Image metas lidar2img:', len(img_metas['lidar2img']), img_metas['lidar2img'][0].shape)
        #     print('Image metas pts_filename:', img_metas['pts_filename'])
        #     print('Save dir:', result_path)
        #
        #     vis_data = dict(
        #         lidar_file_name=file_name,
        #         query = query,
        #         query_pos = query_pos,
        #         reference_points = reference_points,
        #         img_bev_feats = bev_embed)
        #     torch.save(vis_data, osp.join(result_path, 'vis_data.pt'))
        # ##
        return bev_embed, inter_states, init_reference_out, inter_references_out