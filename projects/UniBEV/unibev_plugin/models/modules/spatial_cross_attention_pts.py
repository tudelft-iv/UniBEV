from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttentionPts(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=None,
                 **kwargs
                 ):
        super(SpatialCrossAttentionPts, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                reference_points_lidar=None,
                bev_mask=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_lidar.size(0)
        indexes = []

        # ## debug
        # print('\nIn Spatial Cross Attention VoxelDETR')
        # print('query:', query.size()) # (2, 40000, 512)
        # print('key:', key.size()) # (40000, 2, 512)
        # print('value:', value.size()) # same as key: (40000, 2, 512)
        # print('query pose:', query_pos) # None
        # print('reference points lidar:', reference_points_lidar.size()) # original reference points cam (6, 1, 22500, 4, 2) -> now (4, 2, 40000, 3)
        # ##

        # for i, mask_per_img in enumerate(bev_mask):
        #     index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            # ## debug in the mask_per_img function
            # print('\n'
            #       '\nindex:', i,
            #       '\nmask per img size:', mask_per_img.size(), # (1, 22500, 4)
            #       '\nmask_per_img[0]:', mask_per_img[0].size(), # (22500, 4)
            #       '\nmask_per_img[0].sum(-1):', mask_per_img[0].sum(-1).size(), # (22500)
            #       '\nmask_per_img[0].sum(-1).nonzero():', mask_per_img[0].sum(-1).nonzero().size(), # (4249, 1), get the indexes of the non-zero element
            #       '\nmask_per_img[0].sum(-1).nonzero().squeeze(-1):', mask_per_img[0].sum(-1).nonzero().squeeze(-1).size()) # (4249)
            # ## this is used to get the non-empty/or only related voxels in the bev
        #     indexes.append(index_query_per_img)
        # max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        # queries_rebatch = query.new_zeros(
        #     [bs, self.num_cams, max_len, self.embed_dims])
        # reference_points_rebatch = reference_points_cam.new_zeros(
        #     [bs, self.num_cams, max_len, D, 2])

        # ## debug
        # print('queries_rebatch:', queries_rebatch.size()) # (1, 6, 5364, 256)
        # print('reference_points_rebatch:', reference_points_rebatch.size()) # (1, 6, 5364, 4, 2)
        # ##
        
        # for j in range(bs):
        #     for i, reference_points_per_img in enumerate(reference_points_cam):
        #         index_query_per_img = indexes[i]
        #         queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
        #         reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        # num_cams, l, bs, embed_dims = key.shape

        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        reference_points_lidar = reference_points_lidar.permute(1,2,0,3)
        # ## debug: dimensions which are fed into the attention module:
        # print('\nVariables which are fed into DeformableAttention:')
        # print('query size:', queries_rebatch.size())
        # print('RESHAPED query size:', queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims).size())
        # print('key size (flatten image features):', key.size())
        # print('value size (flatten image features):', value.size())
        # print('reference_points size:', reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2).size())
        # print('spatial_shapes size: ', spatial_shapes.size())
        # print('level start index size:', level_start_index.size())
        # ##

        queries = self.deformable_attention(query=query,
                                            key=key,
                                            value=value,
                                            reference_points=reference_points_lidar,
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index)
        # ## debug: queries after attention module:
        # print('Queries size after attention module:', queries.size())
        # ##

        queries = queries.view(bs, -1, self.embed_dims)

        # ## debug: queries after attention module:
        # print('Queries size after view:', queries.size())
        # ##
        # for j in range(bs):
        #     for i, index_query_per_img in enumerate(indexes):
        #         slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # count = bev_mask.sum(-1) > 0
        # count = count.permute(1, 2, 0).sum(-1)
        # count = torch.clamp(count, min=1.0)
        # slots = slots / count[..., None]
        queries_out = self.output_proj(queries)

        return self.dropout(queries_out) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3DPts(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        # ## debug
        # print('grid_init:', grid_init.size())
        # ##
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        ## debug
        # print('\nIn MSDeformableAttention3D of UniBEV:')
        # print(' query size:', query.size()) # (2, 40000, 512)
        # print(' value size:', value.size()) # (2, 40000, 512)
        # print(' reference_points size:', reference_points.size()) #(2, 40000, 4, 3) original: (bs*self.num_cams, max_len, D, 2)

        ##
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        # ## debug
        # print(' Query:', query.size())
        # print(' Sampling Offsets:', self.sampling_offsets(query))
        # sampling_offsets = self.sampling_offsets(query)
        # sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 3)
        # ##
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        # if reference_points.shape[-1] == 2:
        """
        For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
        After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
        For each referent point, we sample `num_points` sampling points.
        For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
        """
        # z_axis = torch.ones_like(spatial_shapes[...,0])*self.num_points
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

        bs, num_query, num_Z_anchors, xy = reference_points.shape
        # (2, 40000, 4, 3)
        # ## debug
        # print(' sampling offsets:', sampling_offsets.size())
        # print(' offset nomalizer:', offset_normalizer.size())
        # print(' offset nomalizer:', offset_normalizer)
        # ##
        reference_points = reference_points[:, :, None, None, None, :, :]
        sampling_offsets = sampling_offsets / \
            offset_normalizer[None, None, None, :, None, :]
        bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
        sampling_offsets = sampling_offsets.view(
            bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
        # print('reference points2:',reference_points.shape)
        # print('sampling_offsets2:',sampling_offsets.shape)
        sampling_locations = reference_points + sampling_offsets
        bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
        assert num_all_points == num_points * num_Z_anchors

        sampling_locations = sampling_locations.view(
            bs, num_query, num_heads, num_levels, num_all_points, xy)

        # elif reference_points.shape[-1] == 4:
        #     assert False
        # else:
        #     raise ValueError(
        #         f'Last dim of reference_points must be'
        #         f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
