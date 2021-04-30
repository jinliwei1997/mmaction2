model = dict(
    type='VideoTextMatcherE2E',
    backbone1=dict(
        type='ResNet',
        pretrained=None,
        depth=50,
        norm_eval=False),
    backbone2=dict(
        type='BERT',
        pretrained='/mnt/lustre/jinliwei/bert_model',
        freeze=True
    ),
    head=dict(
        type='MILNCEHead',
        temperature=0.05,
    ),
    fp16_enabled=False,
    img_feat_dim=2048,
    text_feat_dim=768,
    feature_dim=256,
    init_std=0.01,
    gather_flag = False
)
train_cfg = None
test_cfg = None
dataset_type = 'Mp4TextDataset'
data_root = ''
data_root_val = ''
ann_file_train = '/mnt/lustre/jinliwei/vatex_process/vatex_train_bert_ch1'
ann_file_val = '/mnt/lustre/jinliwei/vatex_process/vatex_val_bert_ch1'
ann_file_test = '/mnt/lustre/jinliwei/vatex_process/vatex_val_bert_ch1_1000'
mc_cfg = dict(
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit',
         io_backend='memcached',
         **mc_cfg),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        lazy=True),
    dict(type='Resize', scale=(112, 112), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='LoadTexts', sample_mode='number', sample_number=1),
    dict(type='TextTokenize', tokenizer_dir='/mnt/lustre/jinliwei/bert_model'),
    dict(type='Collect', keys=['imgs', 'texts_item'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
val_pipeline = [
    dict(type='DecordInit',
         io_backend='memcached',
         **mc_cfg),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='CenterCrop', crop_size=224, lazy=True),
    dict(type='Resize', scale=(112, 112), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='LoadTexts', sample_mode='number', sample_number=1),
    dict(type='TextTokenize', tokenizer_dir='/mnt/lustre/jinliwei/bert_model'),
    dict(type='Collect', keys=['imgs', 'texts_item'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit',
         io_backend='memcached',
         **mc_cfg),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(type='CenterCrop', crop_size=224, lazy=True),
    dict(type='Resize', scale=(112, 112), keep_ratio=False, lazy=True),
    dict(type='Flip', flip_ratio=0, lazy=True),
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='LoadTexts', sample_mode='number', sample_number=1),
    dict(type='TextTokenize', tokenizer_dir='/mnt/lustre/jinliwei/bert_model'),
    dict(type='Collect', keys=['imgs', 'texts_item'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=48,
    workers_per_gpu=5,
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
        pipeline=test_pipeline)
)

optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=1
)
total_epochs = 100
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=1,
)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
dist_params = dict(backend='nccl',port = 29549)
log_level = 'INFO'
work_dir = './work_dirs/vatex_dm'
load_from = '../ckpt/epoch_80.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters=True