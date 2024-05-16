import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from torch.nn import functional as F
from mmdet3d.core import bbox3d2result
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from mmdet3d.unibev_plugin.models.utils.grid_mask import GridMask
from mmcv.ops import Voxelization
import time
import copy
import numpy as np
import mmdet3d


@DETECTORS.register_module()
class UniBEV(MVXTwoStageDetector):
    """
    UniBEV model:
        a multi-modal fusion model based on BEVFormer_Deformable(cam) and BEVVoxelDetr (lidar).
        using unified bev query to build BEV embedding from image features and point features
        todo
        # temporal information is not applied.
        # video_test_mode (bool): Decide whether to use temporal information during inference.
    Args:
    """

    def __init__(self,
                 use_lidar = True,
                 use_camera = True,
                 use_radar = False,

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

                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        super(UniBEV,
              self).__init__(pts_voxel_layer,
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
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)

        if radar_voxel_layer:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)

        self.fusion_method = pts_bbox_head['transformer'].get('fusion_method', None)

    def extract_img_feat(self, img, img_metas=None):
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
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
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_pts_feat(self, pts):
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

    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.radar_voxelize(radar)

        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        if self.with_pts_backbone:
            x = self.radar_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            if self.with_pts_neck:
                x = self.pts_neck(x)
            return x
        else:
            x = self.radar_middle_encoder(voxel_features, coors, batch_size)
            return [x]

    def extract_feat(self, img, points, radar_points, img_metas=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas) if self.use_camera else None
        pts_feats = self.extract_pts_feat(points) if self.use_lidar else None
        radar_feats = self.extract_radar_feat(radar_points, img_metas) if self.use_radar else None

        return img_feats, pts_feats, radar_feats

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

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
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


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      radar=None,
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
        if self.use_camera:
            assert img is not None
        if self.use_lidar:
            assert points is not None
        if self.use_radar:
            assert radar is not None

        if img is not None:
            len_queue = img.size(1)
            img = img[:, -1, ...]
        else:
            len_queue = 3
            img = None
        img_metas = [each[len_queue - 1] for each in img_metas]
        img_feats, lidar_feats, radar_feats = self.extract_feat(img=img, points=points, radar_points= radar, img_metas=img_metas)

        losses = dict()
        if self.use_lidar == True and self.use_radar == False:
            pts_feats = lidar_feats
        elif self.use_lidar == False and self.use_radar == True:
            pts_feats = radar_feats
        elif self.use_lidar == True and self.use_radar == True:
            raise ValueError('Unsupported Modality Mode: Cam: {}, Lidar:{}, Radar:{}'.format(self.use_camera, self.use_lidar, self.use_radar))
        else:
            pts_feats = None

        outs = self.pts_bbox_head(img_feats, pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses_pts = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None, radar=None, **kwargs):

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        # num_augs = len(points)
        # if num_augs != len(img_metas):
        #     raise ValueError(
        #         'num of augmentations ({}) != num of image meta ({})'.format(
        #             len(points), len(img_metas)))

        img = [img] if img is None else img
        points = [points] if points is None else points
        radar = [radar] if radar is None else radar

        bbox_results, bev_embeds = self.simple_test(
            points[0], img_metas[0], img[0], radar[0], **kwargs)
        return bbox_results

    def simple_test(self, points, img_metas, img=None, radar = None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, lidar_feats, radar_feats = self.extract_feat(img=img, points=points, radar_points=radar, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]

        if self.use_lidar == True and self.use_radar == False:
            pts_feats = lidar_feats
        elif self.use_lidar == False and self.use_radar == True:
            pts_feats = radar_feats
        elif self.use_lidar == True and self.use_radar == True:
            raise ValueError('Unsupported Modality Mode: Cam: {}, Lidar:{}, Radar:{}'.format(self.use_camera, self.use_lidar, self.use_radar))
        else:
            pts_feats = None

        outs = self.pts_bbox_head(img_feats, pts_feats, img_metas)

        bbox_list_head = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_pts = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list_head
        ]

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list, outs['bev_embed']
