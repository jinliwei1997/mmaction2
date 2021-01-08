import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmaction import __version__
from mmaction.apis import train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import collect_env, get_root_logger

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

cfg = dict(
    type = 'VideoTextDataset',
    ann_file = 'test_dataset/annotation',
    data_prefix = 'test_dataset',
    pipeline=[
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
        dict(type='LoadTexts', sample_ratio=0.2),
        dict(type='Collect', keys=['imgs', 'texts','texts_locations'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
)

video_text_dataset = build_dataset(cfg)

result = video_text_dataset[0]
print (result)
