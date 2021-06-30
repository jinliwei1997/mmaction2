from ..registry import RECOGNIZERS
from .base import BaseRecognizer


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

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)
