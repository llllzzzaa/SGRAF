_base_ = [ '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_1x.py']
backend_args = None
load_from = 'path/to/grounding_dino_swin-t_finetune_16xb2_1x_coco_20230921_152544-5f234b20.pth'  # noqa  
lang_model_name = 'path/to/bert-base-uncased'
class_name = ('person',)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)]
                    )
model = dict(
    type='TextDualSpectralGroundingDINOillum',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DualSteramDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        mean_ir=[132.345, 132.345, 132.345],  # IR as 3-channel  input
        std=[58.395, 57.12, 57.375],
        std_ir=[57.375, 57.375, 57.375], 
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=False,
        ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        # visual layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        # text layer config
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        # fusion layer config
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            # query self attention layer
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to text
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            # cross attention layer query to image
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale=0.0, bias=False),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        type='EpochBasedTrainLoop', 
        max_epochs=24, 
        val_interval=1,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),      
    test_cfg=dict(
        type='TestLoop',
        max_per_img=300))

# dataset settings
train_pipeline = [
    dict(type='LoadAlignedImagesFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AlignedImagesRandomFlip', prob=0.5),
    dict(
        type='AlignedImagesResize',
        scale=(640,640),
        keep_ratio=True),
    dict(
        type='AlignedImagesRandomCrop',
        crop_type='absolute_range',
        crop_size=(640,640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='AlignedImagesPad', size=(640,640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackAlignedImagesDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities','scene')) 
]  

test_pipeline = [
    dict(type='LoadAlignedImagesFromFile', backend_args=backend_args),
    dict(type='AlignedImagesResize', scale=(640,640), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackAlignedImagesDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities','scene'))
]
dataset_type = 'DualSpectralDatasetScene'
data_root = 'path/to/LLVIP/' 

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotation_scene_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(
            filter_empty_gt=False, 
            min_size=32             
        ),
        pipeline=train_pipeline,    
        backend_args=backend_args,
        return_classes=True          
    )
)

val_dataloader = dict(
    batch_size=6,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotation_scene_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,  
        backend_args=backend_args,
        return_classes=True     
    )
)
test_dataloader = val_dataloader


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotation_scene_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.0),
        'language_model': dict(lr_mult=0.0)
    }))
# learning policy
max_epochs = 24
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[22],
        gamma=0.1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        save_best='coco/bbox_mAP_50',
        # rule='less',
        interval=1,
        max_keep_ckpts=1,
    )
)

auto_scale_lr = dict(base_batch_size=32)
