# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shiming Wang
# ---------------------------------------------
import time
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
class PPDETRTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        encoder (obj:`ConfigDict`): encoder definition.
            Default: None.
        decoder (obj:`ConfigDict`): decoder definition.
            Default: None.
        embedding_dims (int): embedding dimensions in transformers
            Default: 256.
    """

    def __init__(self,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 **kwargs):
        super(PPDETRTransformer, self).__init__(**kwargs)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        else:
            self.decoder = None


        self.embed_dims = embed_dims
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

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
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                pts_feats,
                object_query_embed,
                bev_h,
                bev_w,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            pts_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
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

    # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs, C_pts, H_pts, W_pts = pts_feats[0].size()
        pts_feats_bev = pts_feats[0].permute(2, 3, 0, 1).view(H_pts*W_pts, bs, C_pts)
        bev_embed = pts_feats_bev

        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points
        # tmp = time.time()
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        ##Visualization Part in Test
        # if self.training is False:
        #     #img_metas = kwargs['img_metas'][0]
        #     out_dir = 'outputs/inference/pp_detr_nus_L_full_val_mini_visualization'  # hardcoded
        #     #pts_path = img_metas['pts_filename']
        #     file_name = '0'
        #     result_path = osp.join(out_dir, file_name)
        #
        #     print('\nMode:', self.training)
        #     print('Query Size:', query.size())
        #     print('Query Pose Size:', query_pos.size())
        #     print('Reference Point:', reference_points.size())
        #     print('Pts BEV features:', bev_embed.size())
        #     # print('Image metas:', img_metas.keys())
        #     # print('Image metas filename:', img_metas['filename'])
        #     # print('Image metas lidar2img:', len(img_metas['lidar2img']), img_metas['lidar2img'][0].shape)
        #     # print('Image metas pts_filename:', img_metas['pts_filename'])
        #     print('Save dir:', result_path)
        #
        #     vis_data = dict(
        #         query=query,
        #         query_pos=query_pos,
        #         reference_points=reference_points,
        #         pts_bev_feats=bev_embed)
        #     torch.save(vis_data, osp.join(result_path, 'vis_data_{}.pt'.format(tmp)))
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

        return bev_embed, inter_states, init_reference_out, inter_references_out
