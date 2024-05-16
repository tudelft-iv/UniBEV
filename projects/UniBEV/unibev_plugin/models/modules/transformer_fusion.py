import copy
import os.path as osp

import fontTools.ttLib
import mmcv.utils.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_, constant_

from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate

from .spatial_cross_attention_img import MSDeformableAttention3DImg
from .spatial_cross_attention_pts import MSDeformableAttention3DPts
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from .decoder import CustomMSDeformableAttention
from mmcv.runner import force_fp32, auto_fp16

class ModalityProjectionModule(BaseModule):
    def __init__(self,
                 embed_dims,
                 with_norm = True,
                 with_residual = True,
                 ):
        super().__init__()
        layers = nn.ModuleList(
           [ nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True)]
        )
        if with_norm:
            layers.append(nn.LayerNorm(embed_dims))

        self.net = nn.Sequential(*layers)
        self.with_residual = with_residual
    def forward(self, x):
        out = self.net(x)
        if self.with_residual:
            return x + out
        else:
            return out

@TRANSFORMER.register_module()
class UniBEVTransformer(BaseModule):
    """Implements the UniBEV transformer, based on Detr3D transformer.
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
                 img_encoder=None,
                 pts_encoder=None,
                 decoder=None,
                 embed_dims=256,
                 use_cams_embeds=True,
                 fusion_method='linear',
                 drop_modality = None,
                 feature_norm = None,
                 spatial_norm = None,
                 use_modal_embeds = None,
                 bev_h = 200,
                 bev_w = 200,
                 dual_queries = False,
                 vis_output = None,
                 cna_constant_init = None,
                 **kwargs):
        super(UniBEVTransformer, self).__init__(**kwargs)

        if img_encoder is not None:
            self.img_bev_encoder = build_transformer_layer_sequence(img_encoder)

        if pts_encoder is not None:
            self.pts_bev_encoder = build_transformer_layer_sequence(pts_encoder)

        self.decoder = build_transformer_layer_sequence(decoder)

        self.dual_queries = dual_queries
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.cna_constant_norm = cna_constant_init
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.use_cams_embeds = use_cams_embeds
        # self.finetune_init_value = finetune_init_value

        self.fusion_method = fusion_method

        if self.fusion_method == 'linear' or self.fusion_method == 'avg':
            self.scale_factor = 1
        elif self.fusion_method == 'cat':
            self.scale_factor = 2  # used to scale up the dimension when concatenate
        else:
            raise ValueError('Unrecognizable fusion method:{}'.format(self.fusion_method))

        self.drop_modality = drop_modality
        self.feature_norm = feature_norm
        self.spatial_norm = spatial_norm
        self.use_modal_embeds = use_modal_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.vis_output = vis_output
        self.init_layers()
    
    @property
    def with_img_bev_encoder(self):
        """bool: Whether the img_bev_encoder exists."""
        return hasattr(self, 'img_bev_encoder') and self.img_bev_encoder is not None

    @property
    def with_pts_bev_encoder(self):
        """bool: Whether the pts_bev_encoder exists."""
        return hasattr(self, 'pts_bev_encoder') and self.pts_bev_encoder is not None

    def init_layers(self):
        """Initialize layers of the UniBEVTransformer, based on DETR3D."""
        if self.feature_norm == 'ChannelNormWeights':
            self.feature_norm_layer = nn.Softmax(dim=0)
            self.pts_channel_weights = nn.Parameter(torch.Tensor(self.embed_dims))
            self.img_channel_weights = nn.Parameter(torch.Tensor(self.embed_dims))
        elif self.feature_norm == 'MLP_ChannelNormWeights':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 2),
                nn.ReLU(inplace=True))
        elif self.feature_norm == 'Leaky_ReLU_MLP_ChannelNormWeights':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 2),
                nn.LeakyReLU(inplace=True))
        elif self.feature_norm == 'ELU_MLP_ChannelNormWeights':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 2),
                nn.ELU(inplace=True))
        elif self.feature_norm == 'Sigmoid_MLP_ChannelNormWeights':
            self.channel_weights_proj = nn.Sequential(
                nn.Linear(self.bev_h * self.bev_w * 2, 2),
                nn.Sigmoid())
        elif self.feature_norm == 'ModalityProjection':
            assert self.fusion_method == 'cat'
            self.c_modal_proj = ModalityProjectionModule(self.embed_dims)
            self.l_modal_proj = ModalityProjectionModule(self.embed_dims)

        if self.spatial_norm == 'SpatialNormWeights':
            self.spatial_norm_layer = nn.Softmax(dim=0)
            self.pts_spatial_weights = nn.Parameter(torch.Tensor(self.bev_h*self.bev_w))
            self.img_spatial_weights = nn.Parameter(torch.Tensor(self.bev_h*self.bev_w))

        if self.with_img_bev_encoder:
            self.img_level_embeds = nn.Parameter(torch.Tensor(
                self.num_feature_levels, self.embed_dims))
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, self.embed_dims))

        if self.with_pts_bev_encoder:
            self.pts_level_embeds = nn.Parameter(torch.Tensor(
                self.num_feature_levels, self.embed_dims))

        if self.use_modal_embeds == 'MLP':
            self.modal_embbeding_mlp = nn.Sequential(
                nn.Linear(2, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True))
        elif self.use_modal_embeds == 'Fixed':
            self.modal_embbeding_C = nn.Parameter(torch.Tensor(self.embed_dims))
            self.modal_embbeding_L = nn.Parameter(torch.Tensor(self.embed_dims))

        self.reference_points = nn.Linear(self.embed_dims * self.scale_factor, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3DPts) or isinstance(m, MSDeformableAttention3DImg) \
                    or isinstance(m, MultiScaleDeformableAttention) or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        if self.with_pts_bev_encoder:
            normal_(self.pts_level_embeds)
        if self.with_img_bev_encoder:
            normal_(self.img_level_embeds)
            normal_(self.cams_embeds)
        if self.feature_norm == 'ChannelNormWeights':
            if self.cna_constant_norm == True:
                constant_(self.pts_channel_weights, 0.5)
                constant_(self.img_channel_weights, 0.5)
            else:
                normal_(self.pts_channel_weights)
                normal_(self.img_channel_weights)
        if self.feature_norm in ('MLP_ChannelNormWeights',
                                 'Leaky_ReLU_MLP_ChannelNormWeights',
                                 'ELU_MLP_ChannelNormWeights',
                                 'Sigmoid_MLP_ChannelNormWeights'):
            xavier_init(self.channel_weights_proj, distribution='uniform', bias=0)
        if self.feature_norm == 'ModalityProjection':
            xavier_init(self.c_modal_proj)
            xavier_init(self.l_modal_proj)
        if self.spatial_norm == 'SpatialNormWeights':
            normal_(self.pts_spatial_weights)
            normal_(self.img_spatial_weights)

        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        if self.use_modal_embeds == 'MLP':
            xavier_init(self.modal_embbeding_mlp, distribution='uniform', bias=0.)
        elif self.use_modal_embeds == 'Fixed':
            normal_(self.modal_embbeding_C)
            normal_(self.modal_embbeding_L)

    def get_probability(self, prob):
        return True if np.random.random() < prob else False

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def _pre_process_img_feats(self, mlvl_img_feats, bev_queries):
        """
        preprocess img features
        """

        img_feat_flatten = []
        img_spatial_shapes = []
        for lvl, feat in enumerate(mlvl_img_feats):
            bs, num_cam, c, h, w = feat.shape
            img_spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.img_level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            img_spatial_shapes.append(img_spatial_shape)
            img_feat_flatten.append(feat)

        img_feat_flatten = torch.cat(img_feat_flatten, 2)
        img_spatial_shapes = torch.as_tensor(img_spatial_shapes, dtype=torch.long, device=bev_queries.device)
        img_level_start_index = torch.cat((img_spatial_shapes.new_zeros((1,)), img_spatial_shapes.prod(1).cumsum(0)[:-1]))

        img_feat_flatten = img_feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        return img_feat_flatten, img_spatial_shapes, img_level_start_index

    def _pre_process_pts_feats(self, mlvl_pts_feats, bev_queries):
        ## process multi-level points features
        pts_feat_flatten = []
        pts_spatial_shapes = []

        for lvl, feat in enumerate(mlvl_pts_feats):

            bs, c, h, w = feat.shape
            pts_spatial_shape = (h, w)
            feat = feat.flatten(2).permute(0, 2, 1)
            # print(' feat size:', feat.size()) # [2, 40000, 512]
            feat = feat + self.pts_level_embeds[None, lvl:lvl + 1, :].to(feat.dtype)
            pts_spatial_shapes.append(pts_spatial_shape)
            pts_feat_flatten.append(feat)

        pts_feat_flatten = torch.cat(pts_feat_flatten, 2)
        pts_spatial_shapes = torch.as_tensor(pts_spatial_shapes, dtype=torch.long, device=bev_queries.device)
        pts_level_start_index = torch.cat((pts_spatial_shapes.new_zeros((1,)), pts_spatial_shapes.prod(1).cumsum(0)[:-1]))

        pts_feat_flatten = pts_feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        return pts_feat_flatten, pts_spatial_shapes, pts_level_start_index

    def multi_modal_fusion(self, img_bev_embed, pts_bev_embed):
        assert self.fusion_method is not None
        if self.fusion_method == 'linear':
            fused_bev_embed = self.c_flag * img_bev_embed + self.l_flag * pts_bev_embed
        elif self.fusion_method == 'avg':
            fused_bev_embed = img_bev_embed * self.c_flag / (self.c_flag + self.l_flag) + pts_bev_embed * self.l_flag / (self.c_flag + self.l_flag)
        elif self.fusion_method == 'cat':
            if self.feature_norm == 'ModalityProjection':
                assert img_bev_embed.shape[-1] == self.embed_dims * 2 and pts_bev_embed.shape[-1] == self.embed_dims * 2
                c_pseudo_flag = 1 - self.c_flag
                l_pseudo_flag = 1 - self.l_flag

                c_true_weights = torch.Tensor([self.c_flag]).expand(self.embed_dims)
                l_psudo_weights = torch.Tensor([l_pseudo_flag]).expand(self.embed_dims)
                l_true_weights = torch.Tensor([self.l_flag]).expand(self.embed_dims)
                c_pseudo_weights = torch.Tensor([c_pseudo_flag]).expand(self.embed_dims)

                img_flags = torch.cat((c_true_weights, l_psudo_weights)).cuda()
                pts_flags = torch.cat((c_pseudo_weights, l_true_weights)).cuda()

                fused_bev_embed = img_bev_embed * img_flags + pts_bev_embed * pts_flags
            else:
                fused_bev_embed = torch.cat((img_bev_embed * self.c_flag, pts_bev_embed * self.l_flag), -1)
        else:
            raise NotImplementedError

        if self.use_modal_embeds == 'MLP':
            modal_status = torch.Tensor([self.c_flag, self.l_flag]).cuda()
            modal_embedding = self.modal_embbeding_mlp(modal_status)
            fused_bev_embed += modal_embedding
        elif self.use_modal_embeds == 'Fixed':
            modal_embedding = self.c_flag * self.modal_embbeding_C + self.l_flag * self.modal_embbeding_L
            fused_bev_embed += modal_embedding

        return fused_bev_embed

    def channel_feature_norm(self, img_bev_embed, pts_bev_embed):
        # (bs, bev_h * bev_w, embed_dims)
        if img_bev_embed is None:
            img_bev_embed = torch.zeros_like(pts_bev_embed)
        elif pts_bev_embed is None:
            pts_bev_embed = torch.zeros_like(img_bev_embed)
        vis_data = None
        if self.feature_norm == 'ChannelNormWeights':
            channel_weight_list = []
            channel_weight_list.append(self.img_channel_weights.unsqueeze(0))
            channel_weight_list.append(self.pts_channel_weights.unsqueeze(0))
            feature_weights = torch.cat(channel_weight_list, 0)
            if self.c_flag == 1 and self.l_flag == 1:
                channel_weights_norm = self.feature_norm_layer(feature_weights)
                img_norm_weights = channel_weights_norm[0]
                pts_norm_weights = channel_weights_norm[1]
            else:
                img_norm_weights = self.feature_norm_layer(feature_weights[0:1])[0]
                pts_norm_weights = self.feature_norm_layer(feature_weights[1:2])[0]

            img_bev_embed = img_bev_embed * img_norm_weights
            pts_bev_embed = pts_bev_embed * pts_norm_weights
            if self.vis_output:
                vis_data = dict(
                    feature_weights = feature_weights,
                    channel_weights_norm = channel_weights_norm,
                    img_norm_weights = img_norm_weights,
                    pts_norm_weights = pts_norm_weights
                )
        elif self.feature_norm in ('MLP_ChannelNormWeights',
                                   'Leaky_ReLU_MLP_ChannelNormWeights',
                                   'ELU_MLP_ChannelNormWeights',
                                   'Sigmoid_MLP_ChannelNormWeights'):
            input_bev_feats = torch.cat([img_bev_embed, pts_bev_embed], dim=1).permute(0,2,1)
            multi_modal_channel_weights = self.channel_weights_proj(input_bev_feats)

            if self.c_flag == 1 and self.l_flag ==1:
                multi_modal_channel_norm_weights = F.softmax(multi_modal_channel_weights, dim=-1)
                img_norm_weights = multi_modal_channel_norm_weights[:,:,0]
                pts_norm_weights = multi_modal_channel_norm_weights[:,:,1]

            else:
                img_norm_weights = F.softmax(multi_modal_channel_weights[:,:, :1], dim=-1).squeeze(-1)
                pts_norm_weights = F.softmax(multi_modal_channel_weights[:,:, 1:], dim=-1).squeeze(-1)

            img_bev_embed = img_bev_embed * img_norm_weights[:, None, :]
            pts_bev_embed = pts_bev_embed * pts_norm_weights[:, None, :]

            if self.vis_output:
                vis_data = dict(
                    multi_modal_channel_weights = multi_modal_channel_weights,
                    multi_modal_channel_norm_weights = multi_modal_channel_norm_weights,
                    img_norm_weights = img_norm_weights,
                    pts_norm_weights = pts_norm_weights
                )
        elif self.feature_norm == 'ModalityProjection':
            pseudo_pts_bev_embed = self.l_modal_proj(img_bev_embed)
            pseudo_img_bev_embed = self.c_modal_proj(pts_bev_embed)

            img_bev_embed = torch.cat([img_bev_embed, pseudo_pts_bev_embed], dim=-1)
            pts_bev_embed = torch.cat([pseudo_img_bev_embed, pts_bev_embed], dim=-1)

            if self.vis_output:
                vis_data = dict(
                    pseudo_pts_bev_embed = pseudo_pts_bev_embed,
                    pseudo_img_bev_embed = pseudo_img_bev_embed,
                )

        return img_bev_embed, pts_bev_embed, vis_data

    def spatial_feature_norm(self, img_bev_embed, pts_bev_embed):
        # (bs, bev_h * bev_w, embed_dims)
        vis_data = None
        if self.spatial_norm == 'SpatialNormWeights':
            spatial_weight_list = []
            spatial_weight_list.append(self.img_spatial_weights.unsqueeze(0))
            spatial_weight_list.append(self.pts_spatial_weights.unsqueeze(0))
            spatial_weights = torch.cat(spatial_weight_list, 0)

            if self.c_flag == 1 and self.l_flag == 1:
                spatial_weights_norm = self.spatial_norm_layer(spatial_weights)
                img_spatial_norm_weights = spatial_weights_norm[0]
                pts_spatial_norm_weights = spatial_weights_norm[1]
            else:
                img_spatial_norm_weights = self.spatial_norm_layer(spatial_weights[:1])[0]
                pts_spatial_norm_weights = self.spatial_norm_layer(spatial_weights[1:])[0]

            img_bev_embed = img_bev_embed * img_spatial_norm_weights[None,:,None]
            pts_bev_embed = pts_bev_embed * pts_spatial_norm_weights[None,:,None]

            if self.vis_output:
                vis_data = dict(
                    spatial_weights = spatial_weights,
                    spatial_weights_norm = spatial_weights_norm,
                    img_spatial_norm_weights = img_spatial_norm_weights,
                    pts_spatial_norm_weights = pts_spatial_norm_weights
                )
        return img_bev_embed, pts_bev_embed, vis_data

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                img_mlvl_feats,
                pts_mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
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
        self.l_flag = 1
        self.c_flag = 1
        if self.drop_modality is not None and self.training is True:
            if isinstance(self.drop_modality, dict):
                dropout_prob = self.drop_modality['dropout_prob']
                lidar_prob = self.drop_modality['lidar_prob']
            elif isinstance(self.drop_modality, float):
                dropout_prob = self.drop_modality
                lidar_prob = self.drop_modality
            else:
                raise ValueError('Unrecognized type: {}'.format(type(self.drop_modality)))
            v_flag = self.get_probability(dropout_prob)
            if v_flag:
                self.l_flag = self.get_probability(lidar_prob)*1
                self.c_flag = 1 - self.l_flag
            # print('dropout_prob:', dropout_prob)
            # print('lidar_prob:', lidar_prob)
            # print('v_flag, l_flag, c_flag:', v_flag, self.l_flag, self.c_flag)

        if img_mlvl_feats is None:
            self.c_flag = 0
            bs = pts_mlvl_feats[0].size(0)
        elif pts_mlvl_feats is None:
            self.l_flag = 0
            bs = img_mlvl_feats[0].size(0)
        else:
            bs = img_mlvl_feats[0].size(0)
        # print('l_flag, c_flag:', self.l_flag, self.c_flag)
        if bev_pos is not None:
            bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        if self.dual_queries:
            assert isinstance(bev_queries, list)
            bev_queries_img = bev_queries[0].unsqueeze(1).repeat(1, bs, 1)
            bev_queries_pts = bev_queries[1].unsqueeze(1).repeat(1, bs, 1)
        else:
            bev_queries_img = bev_queries_pts = bev_queries.unsqueeze(1).repeat(1, bs, 1)

        if img_mlvl_feats is not None:
            img_feat_flatten, img_spatial_shapes, img_level_start_index = self._pre_process_img_feats(img_mlvl_feats, bev_queries_img)
            img_bev_embed = self.img_bev_encoder(
                bev_queries_img,
                img_feat_flatten,
                img_feat_flatten,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                spatial_shapes = img_spatial_shapes,
                level_start_index = img_level_start_index,
                **kwargs) # encoder.batch_first = True: (bs, bev_h*bev_w, embed_dims)
        else:
            img_bev_embed = None

        if pts_mlvl_feats is not None:
            pts_feat_flatten, pts_spatial_shapes, pts_level_start_index = self._pre_process_pts_feats(pts_mlvl_feats, bev_queries_pts)
            pts_bev_embed = self.pts_bev_encoder(
                bev_queries_pts,
                pts_feat_flatten,
                pts_feat_flatten,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_pos=bev_pos,
                spatial_shapes=pts_spatial_shapes,
                level_start_index=pts_level_start_index,
                **kwargs) # encoder.batch_first = True: (bs, bev_h*bev_w, embed_dims)
        else:
            pts_bev_embed = None

        if self.vis_output is not None:
            vis_data = dict(
                ori_img_bev_embed=img_bev_embed.clone(),
                ori_pts_bev_embed=pts_bev_embed.clone(),
            )
        img_bev_embed, pts_bev_embed, vis_data_channel = self.channel_feature_norm(img_bev_embed, pts_bev_embed)
        img_bev_embed, pts_bev_embed, vis_data_spatial = self.spatial_feature_norm(img_bev_embed, pts_bev_embed)

        fused_bev_embed = self.multi_modal_fusion(img_bev_embed, pts_bev_embed)

        query_pos, query = torch.split(object_query_embed, self.embed_dims * self.scale_factor, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        fused_bev_embed = fused_bev_embed.permute(1, 0, 2)

        ## Visualization of features
        if self.training is False and self.vis_output is not None:
            assert isinstance(self.vis_output, dict)
            outdir = self.vis_output['outdir']
            pts_path = kwargs['img_metas'][0]['pts_filename']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            result_path = osp.join(outdir, file_name)
            mmcv.utils.path.mkdir_or_exist(result_path)
            if vis_data_channel is not None:
                vis_data.update(vis_data_channel)
            if vis_data_spatial is not None:
                vis_data.update(vis_data_spatial)

            for key in self.vis_output['keys'] + self.vis_output['special_keys']:
                vis_data[key] = locals()[key]
            for attr in self.vis_output['attrs']:
                vis_data[attr] = getattr(self, attr)

            vis_data.update(dict(lidar_file_name=file_name))
            torch.save(vis_data, osp.join(result_path, 'vis_data.pt'))
        ##
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=fused_bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return fused_bev_embed, inter_states, init_reference_out, inter_references_out
