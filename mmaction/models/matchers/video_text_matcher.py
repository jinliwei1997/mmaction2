from ..registry import MATCHERS
from .base import BaseMatcher
import torch.nn as nn
from .. import builder
from mmcv.cnn import normal_init

@MATCHERS.register_module()
class VideoTextMatcher(BaseMatcher):
    """VideoTextMatcher model framework."""
    def __init__(self,
        backbone1,
        backbone2,
        head,
        neck=None,
        train_cfg=None,
        test_cfg=None,
        fp16_enabled=False,
        img_in_channels = 2048,
        text_in_channels = 768,
        hidden_state_channels = 256,
        init_std = 0.01):
        super(VideoTextMatcher, self).__init__(backbone1,backbone2,head,train_cfg,test_cfg,fp16_enabled)

        self.img_in_channels = img_in_channels
        self.text_in_channels = text_in_channels
        self.hidden_state_channels = hidden_state_channels
        self.init_std = init_std

        self.img_mlp = nn.Sequential(nn.Linear(img_in_channels, self.hidden_state_channels * 2), nn.BatchNorm1d(self.hidden_state_channels * 2), nn.ReLU(), nn.Linear(self.hidden_state_channels * 2, self.hidden_state_channels))
        self.text_mlp = nn.Sequential(nn.Linear(text_in_channels, self.hidden_state_channels * 2), nn.BatchNorm1d(self.hidden_state_channels * 2), nn.ReLU(), nn.Linear(self.hidden_state_channels * 2, self.hidden_state_channels))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.init_mlp_weights()

    def init_mlp_weights(self):
        """Initialize the model network weights."""
        for layer in self.img_mlp:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.text_mlp:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)

    def forward(self, imgs, texts_item, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, texts_item)

        return self.forward_test(imgs, texts_item)

    def extract_v_feat(self, imgs, N):
        x = self.backbone1(imgs)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.reshape((N, -1) + x.shape[1:])
        x = self.consensus(x)
        x = x.squeeze(1)
        # dropout
        x = x.view(x.size(0), -1)
        x = self.img_mlp(x)
        return x

    def extract_t_feat(self, texts, N):
        x = self.backbone2(texts)
        x = self.text_mlp(texts)
        return x

    def forward_train(self, imgs, texts_item):

        N = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        vis_feat = self.extract_v_feat(imgs,N)
        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        text_feat = self.extract_t_feat(texts_item)

        if self.neck is not None:
            vis_feat, text_feat = self.neck(vis_feat, text_feat)

        loss = self.head(vis_feat,text_feat, N)

        return loss

    def forward_test(self, imgs, texts_item):
        """Defines the computation performed at every call when training."""
        N = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        vis_feat = self.extract_v_feat(imgs, N)
        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        text_feat = self.extract_t_feat(texts_item)

        if self.neck is not None:
            vis_feat, text_feat = self.neck(vis_feat, text_feat)

        loss = self.head(vis_feat, text_feat, N)

        return loss

    def forward_gradcam(self, audios):
        raise NotImplementedError

    def train_step(self, data_batch, optimizer, **kwargs):
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
        imgs = data_batch['imgs']
        texts_item = data_batch['texts_item']
        losses = self(imgs, texts_item)

        loss, log_vars = self._parse_losses(losses)
        log_vars['acc'] = 0
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        imgs = data_batch['imgs']
        texts_item = data_batch['texts_item']

        losses = self(imgs, texts_item)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
