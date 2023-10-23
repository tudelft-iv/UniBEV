# Joint training process based on respectively trained 2D and 3D backbone
# BEVFusion-PointPillars
# total in 12 epochs

hostname = 'hpc'
eval_interval = 3
samples_per_gpu = 2
workers_per_gpu = 8
max_epochs = 36
save_interval = 6
log_interval = 10
fusion_method = 'linear'

dataset_type = 'NuScenesDataset_UniQuery'
data_root = 'data/nuscenes/'
sub_dir = 'mmdet3d_bevformer/'
train_ann_file = sub_dir + 'nuscenes_infos_temporal_train.pkl'
val_ann_file = sub_dir + 'nuscenes_infos_temporal_val.pkl'
work_dir = './outputs/train/uniquery_detr_cam_256_pip4_layer_3_nus_C_full'

load_from = 'checkpoints/bevformer/r101_dcn_fcos3d_pretrain.pth'
if hostname == 'iv-mind':
    load_from = 'remote_mounted/' + load_from

resume_from = None
plugin = True
plugin_dir = 'mmdet3d/bevformer_plugin/'

## nuscenes and pointpillars setting
point_cloud_range = [-54, -54, -5, 54, 54, 3]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
file_client_args = dict(backend='disk')
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

### model settings
img_scale = (1600, 900)
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
if fusion_method == 'linear':
    dec_scale_factor = 1
elif fusion_method == 'avg':
    dec_scale_factor = 1
elif fusion_method == 'cat':
    dec_scale_factor = 2

_encoder_layers_ = 3
_num_levels_ = 1
_num_points_in_pillar_cam_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 3 # each sequence contains `queue_length` frames.
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

runner = dict(type='EpochBasedRunner',
              max_epochs=max_epochs)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3DBEVFusion', class_names=class_names), ## which DefaultFormat
    dict(type='CustomCollect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d']) ## which data collection
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3DBEVFusion',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + train_ann_file,
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            bev_size=(bev_h_, bev_w_),
            queue_length=queue_length,
            box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + val_ann_file,
        load_interval=1,
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + val_ann_file,
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        test_mode=True,
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

model = dict(
    type='UniQueryDETRFusion',
    use_grid_mask=True,
    use_lidar=input_modality['use_lidar'],
    use_radar=input_modality['use_radar'],
    use_camera=input_modality['use_camera'],
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True, # using checkpoint to save GPU memory
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='UniQueryDETRFusion_Head',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='UniQueryTransformer',
            embed_dims=_dim_,
            fusion_method=fusion_method,
            img_encoder=dict(
                type='UniqueryDETR_ImgEncoder',
                num_layers=_encoder_layers_,
                pc_range=point_cloud_range,
                num_points_in_pillar=_num_points_in_pillar_cam_,
                return_intermediate=False,
                transformerlayers=dict(
                    type='UniQueryDETR_ImgLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttentionUniQueryImg',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3DUniQueryImg',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                    ),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_ * dec_scale_factor,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_ * dec_scale_factor,
                            num_levels=1),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_ * dec_scale_factor,
                    ),
                    feedforward_channels=_ffn_dim_ * dec_scale_factor,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        assigner=dict(
            type='HungarianAssigner3DBEVFormer',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1CostBEVFormer', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

evaluation = dict(interval=eval_interval, pipeline=eval_pipeline)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            # 'pts_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=125,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# runtime settings
total_epochs = max_epochs

checkpoint_config = dict(interval=save_interval)
log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
custom_hooks = [
    dict(type='CheckpointLateStageHook',
         start=21,
         priority=60)
]
workflow = [('train', 1), ('val', 1)]

