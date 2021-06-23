from ..registry import RECOGNIZERS
from .. import builder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.runner import auto_fp16

from collections import OrderedDict

import numpy as np


@RECOGNIZERS.register_module()
class RecognizerCo(nn.Module):
    """
        2D recognizer model framework for co-teaching.
        Reference: https://arxiv.org/pdf/1804.06872.pdf
    """

    def __init__(
            self,
            backbone1,
            backbone2,
            cls_head1,
            cls_head2,
            neck=None,
            train_cfg=None,
            test_cfg=None,
            tk=10,
            c=1,
            tau=1.0 * 0.3,
            inverse=False,
            min_rate=0.5,
            log_file=None,
    ):
        super().__init__()
        self.backbone1 = builder.build_backbone(backbone1)
        self.backbone2 = builder.build_backbone(backbone2)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.cls_head1 = builder.build_head(cls_head1)
        self.cls_head2 = builder.build_head(cls_head2)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.aux_info = []
        if train_cfg is not None and "aux_info" in train_cfg:
            self.aux_info = train_cfg["aux_info"]

        self.init_weights()

        self.fp16_enabled = False  # might be changed

        self.tk = tk
        self.c = c
        self.tau = tau
        self.inverse = inverse
        self.min_rate = min_rate

        if log_file is not None:
            self.log_file = open(log_file, "w", encoding="utf-8")

    def init_weights(self):
        """Initialize the model network weights."""
        self.backbone1.init_weights()
        self.backbone2.init_weights()
        self.cls_head1.init_weights()
        self.cls_head2.init_weights()
        if hasattr(self, "neck"):
            self.neck.init_weights()

    @auto_fp16()
    def extract_feat(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        feat1 = self.backbone1(imgs)
        feat2 = self.backbone2(imgs)
        return feat1, feat2

    def average_clip(self, cls_score, num_segs=1):
        """Averaging class score over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            torch.Tensor: Averaged class score.
        """
        if "average_clips" not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg["average_clips"]
        if average_clips not in ["score", "prob", None]:
            raise ValueError(
                f"{average_clips} is not supported. "
                f"Currently supported ones are "
                f'["score", "prob", None]'
            )

        if average_clips is None:
            return cls_score

        batch_size = cls_score.shape[0]
        cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        if average_clips == "prob":
            cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == "score":
            cls_score = cls_score.mean(dim=1)

        return cls_score

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
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, imgs, label=None, return_loss=True, epoch=1, idx=None, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get("gradcam", False):
            del kwargs["gradcam"]
            return self.forward_gradcam(imgs)
        if return_loss:
            if label is None:
                raise ValueError("Label should not be None.")
            return self.forward_train(imgs, label, epoch, idx, **kwargs)

        return self.forward_test(imgs)

    def train_step(self, data_batch, optimizer, epoch=1, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
            epoch (int): runner epoch

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        imgs = data_batch["imgs"]
        label = data_batch["label"]
        idx = data_batch["idx"]

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, return_loss=True, epoch=epoch, idx=idx, **aux_info)
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

        losses = self(imgs, label, return_loss=True, **aux_info)  # return_loss=False?

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))),
        )

        return outputs

    def forward_train(self, imgs, labels, epoch=1, idx=0, **kwargs):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        feat1, feat2 = self.extract_feat(imgs)
        if hasattr(self, "neck"):
            feat1 = [
                each.reshape((-1, num_segs) + each.shape[1:])
                    .transpose(1, 2)
                    .contiguous()
                for each in feat1
            ]
            feat1, _ = self.neck(feat1, labels.squeeze())
            feat1 = feat1.squeeze(2)

            feat2 = [
                each.reshape((-1, num_segs) + each.shape[1:])
                    .transpose(1, 2)
                    .contiguous()
                for each in feat2
            ]
            feat2, _ = self.neck(feat2, labels.squeeze())
            feat2 = feat2.squeeze(2)
            num_segs = 1

        cls_score1 = self.cls_head1(feat1, num_segs)
        cls_score2 = self.cls_head2(feat2, num_segs)

        gt_labels = labels.squeeze()

        with torch.no_grad():
            loss_cls1 = F.cross_entropy(cls_score1, gt_labels, reduction="none")
            loss_cls2 = F.cross_entropy(cls_score2, gt_labels, reduction="none")

        ind_1_sorted = torch.argsort(loss_cls1)
        ind_2_sorted = torch.argsort(loss_cls2)

        # remember_rate = 1 - forget_rate
        if not self.inverse:
            remember_rate = 1 - self.tau * min(np.power(epoch, self.c) / self.tk, 1)
        else:
            remember_rate = max(1 / np.sqrt(epoch + 1), self.min_rate)
        num_remember = int(remember_rate * len(ind_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]

        if self.log_file and dist.get_rank() == 0 and np.random.rand() < 0.01:
            idx = np.array([idx1["idx"] for idx1 in idx])
            idx = idx[ind_1_sorted.cpu().numpy()]
            pos_idx = idx[:num_remember]
            neg_idx = idx[num_remember:]
            self.log_file.write(
                f"epoch {epoch}, pos {len(pos_idx)}, neg {len(neg_idx)}\n"
            )
            self.log_file.write(",".join(str(pi) for pi in pos_idx) + "\n")
            self.log_file.write(",".join(str(ni) for ni in neg_idx) + "\n")
            self.log_file.flush()

        # exchange
        loss_1_update = self.cls_head1.loss(
            cls_score1[ind_2_update], gt_labels[ind_2_update], **kwargs
        )  # 'top1_acc', 'top5_acc', 'loss_cls'
        loss_2_update = self.cls_head2.loss(
            cls_score2[ind_1_update], gt_labels[ind_1_update], **kwargs
        )

        losses["top1_acc1"] = loss_1_update["top1_acc"]
        losses["top1_acc2"] = loss_2_update["top1_acc"]
        losses["top5_acc1"] = loss_1_update["top5_acc"]
        losses["top5_acc2"] = loss_2_update["top5_acc"]
        losses["loss_cls1"] = loss_1_update["loss_cls"]
        losses["loss_cls2"] = loss_2_update["loss_cls"]
        losses["remember_rate"] = torch.tensor(remember_rate).cuda()
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        # average loss_cls for two nets
        batches = imgs.shape[0]

        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        feat1, feat2 = self.extract_feat(imgs)
        if hasattr(self, "neck"):
            feat1 = [
                each.reshape((-1, num_segs) + each.shape[1:])
                    .transpose(1, 2)
                    .contiguous()
                for each in feat1
            ]
            feat1, loss_aux = self.neck(feat1)
            feat1 = feat1.squeeze(2)
            losses.update(loss_aux)

            feat2 = [
                each.reshape((-1, num_segs) + each.shape[1:])
                    .transpose(1, 2)
                    .contiguous()
                for each in feat2
            ]
            feat2, loss_aux = self.neck(feat2)
            feat2 = feat2.squeeze(2)
            losses.update(loss_aux)

            num_segs = 1

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score1 = self.cls_head1(feat1, num_segs)
        cls_score2 = self.cls_head2(feat2, num_segs)

        assert (
                cls_score1.size()[0] % batches == 0 and cls_score2.size()[0] % batches == 0
        )
        # calculate num_crops automatically
        cls_score1 = self.average_clip(cls_score1, cls_score1.size()[0] // batches)
        cls_score2 = self.average_clip(cls_score2, cls_score2.size()[0] // batches)

        return (cls_score1 + cls_score2) / 2

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        feat1, feat2 = self.extract_feat(imgs)
        if hasattr(self, "neck"):
            feat1 = [
                each.reshape((-1, num_segs) + each.shape[1:])
                    .transpose(1, 2)
                    .contiguous()
                for each in feat1
            ]
            feat1, loss_aux = self.neck(feat1)
            feat1 = feat1.squeeze(2)
            # losses.update(loss_aux)

            feat2 = [
                each.reshape((-1, num_segs) + each.shape[1:])
                    .transpose(1, 2)
                    .contiguous()
                for each in feat2
            ]
            feat2, loss_aux = self.neck(feat2)
            feat2 = feat2.squeeze(2)
            # losses.update(loss_aux)

            num_segs = 1

        outs = (self.cls_head1(feat1, num_segs), self.cls_head2(feat2, num_segs))
        return outs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)