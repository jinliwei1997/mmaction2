from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from .. import builder

class BaseMatcher(nn.Module, metaclass=ABCMeta):


    def __init__(self, backbone1, backbone2, head, neck=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.backbone1 = builder.build_backbone(backbone1)
        self.backbone2 = builder.build_backbone(backbone2)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None
        self.head = builder.build_head(head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # aux_info is the list of tensor names beyond 'imgs' and 'label' which
        # will be used in train_step and val_step, data_batch should contain
        # these tensors
        self.aux_info = []
        if train_cfg is not None and 'aux_info' in train_cfg:
            self.aux_info = train_cfg['aux_info']

        self.init_weights()

        self.fp16_enabled = False

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone1.init_weights()
        self.backbone2.init_weights()
        self.head.init_weights()
        if hasattr(self, 'neck'):
            self.neck.init_weights()

    @auto_fp16()
    def extract_feat(self, imgs):
        pass

    @abstractmethod
    def forward_train(self, imgs, texts, **kwargs):
        """Defines the computation performed at every call when training."""

    @abstractmethod
    def forward_test(self, imgs, texts,  **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, imgs, texts, return_loss=True, **kwargs):
        pass

    def train_step(self, data_batch, optimizer, **kwargs):
        pass

    def val_step(self, data_batch, optimizer, **kwargs):
        pass