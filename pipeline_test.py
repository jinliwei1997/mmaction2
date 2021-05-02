import os.path as osp
import time
import warnings
import mmcv
import torch
from mmaction.models import build_model
from mmaction.datasets import build_dataset
from tqdm import trange

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

mc_cfg = dict(
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')

cfg = dict(
    type = 'Mp4Word2VecDataset',
    ann_file = '/mnt/lustre/jinliwei/bili_full/bili_anno_vec_train',
    data_prefix ='/mnt/lustre/share_data/bilibili/sensebee_datalist_32109',
    pipeline = [
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
        dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
        dict(type='Flip', flip_ratio=0.5, lazy=True),
        dict(type='Fuse'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='LoadWord2Vec'),
        dict(type='Collect', keys=['imgs', 'word2vec', 'weight'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'word2vec', 'weight'])
    ]
)

bili_full_dataset = build_dataset(cfg)

for i in range(10):
    d = bili_full_dataset[i]
    print(d)

