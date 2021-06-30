from ..registry import RECOGNIZERS
from .. import builder

from .base import BaseRecognizer
import torch.nn as nn
import torch
import torch.distributed as dist

from abc import ABCMeta, abstractmethod
from collections import OrderedDict

@RECOGNIZERS.register_module()
class RecognizerSelfTraining(nn.Module):
    """2D recognizer model framework for self-training."""

    def __init__(
            self,
            teacher_backbone,
            student_backbone,
            teacher_cls_head,
            student_cls_head,
            distill_head,
            neck=None,
            train_cfg=None,
            test_cfg=None,
    ):
        super().__init__()
        self.teacher_backbone = builder.build_backbone(teacher_backbone)
        self.student_backbone = builder.build_backbone(student_backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.teacher_cls_head = builder.build_head(teacher_cls_head)
        self.student_cls_head = builder.build_head(student_cls_head)
        self.distill_head = builder.build_head(distill_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.aux_info = []
        if train_cfg is not None and "aux_info" in train_cfg:
            self.aux_info = train_cfg["aux_info"]

        self.init_weights()

        self.fp16_enabled = False  # might be changed

    def init_weights(self):
        """Initialize the model network weights."""
        self.teacher_backbone.init_weights()
        self.student_backbone.init_weights()
        self.teacher_cls_head.init_weights()
        self.student_cls_head.init_weights()
        if hasattr(self, "neck"):
            self.neck.init_weights()

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        with torch.no_grad():
            x = self.teacher_backbone(imgs)
            teacher_cls_score = self.teacher_cls_head(x, num_segs)

        x = self.student_backbone(imgs)
        student_cls_score = self.student_cls_head(x, num_segs)

        gt_labels = labels.squeeze()

        loss_cls = self.distill_head(teacher_cls_score, student_cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.student_backbone(imgs)
        student_cls_score = self.student_cls_head(x, num_segs)

        assert student_cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(student_cls_score,
                                      student_cls_score.size()[0] // batches)

        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if hasattr(self, 'neck'):
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        outs = (self.cls_head(x, num_segs), )
        return outs

    def train_step(self, data_batch, optimizer, epoch=1, **kwargs):

        imgs = data_batch["imgs"]
        label = data_batch["label"]

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, return_loss=True, **aux_info)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))),
        )

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        imgs = data_batch["imgs"]
        label = data_batch["label"]

        aux_info = {}
        for item in self.aux_info:
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, return_loss=False, **aux_info)  # return_loss=False?

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))),
        )

        return outputs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)

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
