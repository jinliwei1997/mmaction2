model = dict(
    type='VideoTextMatcher',
    backbone1=dict(
        type='ResNet',
        pretrained=None,
        depth=50,
        norm_eval=False),
    backbone2=dict(
        type='BERT',
        pretrained='/mnt/lustre/jinliwei/bert_model'
    ),
    head=dict(
        type='ContrastiveHead',
        img_in_channels=2048,
        text_in_channels=768,
        hidden_state_channels=768,
        temperature=0.1,
        init_std=0.01))
train_cfg = None
test_cfg = None
dataset_type = 'VideoTextDataset'
data_root = 'test_dataset'
data_root_val = 'test_dataset'
ann_file_train = 'test_dataset/annotation'
ann_file_val = 'test_dataset/annotation'
ann_file_test = 'test_dataset/annotation'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=4),
    dict(
        type='RawFrameDecode'
    ),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='LoadTexts', sample_mode='number', sample_number=4),
    dict(type='TextTokenize', tokenizer_dir='/mnt/lustre/jinliwei/bert_model'),
    dict(type='Collect', keys=['imgs', 'texts_item'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(
        type='RawFrameDecode'
    ),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='LoadTexts', sample_mode='number', sample_number=5),
    dict(type='TextTokenize', tokenizer_dir='/mnt/lustre/jinliwei/bert_model'),
    dict(type='Collect', keys=['imgs', 'texts_item'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(
        type='RawFrameDecode'
    ),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='LoadTexts', sample_mode='number', sample_number=2),
    dict(type='TextTokenize', tokenizer_dir='/mnt/lustre/jinliwei/bert_model'),
    dict(type='Collect', keys=['imgs', 'texts_item'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/test/'
load_from = None
resume_from = None
workflow = [('train', 1)]
