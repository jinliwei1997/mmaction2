from ..registry import MATCHERS
from .base import BaseMatcher
import torch.nn as nn
from .. import builder
from mmcv.cnn import normal_init
import torch.distributed as dist
import torch
from collections import OrderedDict

@MATCHERS.register_module()
class VideoAudioTextMatcherE2E(nn.Module):
    """VideoTextMatcher model framework."""
    def __init__(self,
        v_backbone,
        t_backbone,
        head,
        neck = None,
        train_cfg = None,
        test_cfg = None,
        img_feat_dim = 2048,
        text_feat_dim = 768,
        feature_dim = 256,
        init_std = 0.01,
        use_text_mlp = True):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.v_backbone = builder.build_backbone(v_backbone)
        self.t_backbone = builder.build_backbone(t_backbone)
        self.head = builder.build_head(head)
        self.img_feat_dim = img_feat_dim
        self.text_feat_dim = text_feat_dim
        self.feature_dim = feature_dim
        self.init_std = init_std

        self.img_subtitle_mlp = nn.Sequential(nn.Linear(img_feat_dim + text_feat_dim, self.feature_dim * 2), nn.BatchNorm1d(self.feature_dim * 2), nn.ReLU(), nn.Linear(self.feature_dim * 2, self.feature_dim))
        self.text_mlp = nn.Sequential(nn.Linear(text_feat_dim, self.feature_dim * 2), nn.BatchNorm1d(self.feature_dim * 2), nn.ReLU(), nn.Linear(self.feature_dim * 2, self.feature_dim))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.use_text_mlp = use_text_mlp
        self.init_weights()

    def init_weights(self):
        """Initialize the model network weights."""
        self.v_backbone.init_weights()
        self.t_backbone.init_weights()
        self.head.init_weights()
        self.init_mlp_weights()

    def init_mlp_weights(self):
        """Initialize the model network weights."""
        for layer in self.img_subtitle_mlp:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.text_mlp:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)

    def encoder_v(self, imgs, N):
        x = self.v_backbone(imgs)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.reshape((N, -1) + x.shape[1:])
        x = x.mean(dim=1, keepdim=True)
        x = x.squeeze(1)
        # dropout
        x = x.view(x.size(0), -1)
        return x

    def encoder_t(self, texts):
        x = self.t_backbone(texts)
        if self.use_text_mlp:
            x = self.text_mlp(x)
        return x

    def forward(self, imgs, subtitle_texts_item, texts_item, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, subtitle_texts_item, texts_item)

        return self.forward_test(imgs, subtitle_texts_item, texts_item)

    def forward_train(self, imgs, subtitle_texts_item, texts_item):

        N = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        v_feat = self.encoder_v(imgs, N)

        for key in subtitle_texts_item:
            subtitle_texts_item[key] = subtitle_texts_item[key].reshape((-1,) + subtitle_texts_item[key].shape[2:])
        s_feat = self.encoder_t(subtitle_texts_item)

        v_s_feat = nn.functional.normalize(self.img_subtitle_mlp(torch.cat((v_feat, s_feat), dim=1)), dim=1)
        v_s_feat = torch.cat(GatherLayer.apply(v_s_feat), dim=0)

        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        t_feat = nn.functional.normalize(self.encoder_t(texts_item), dim=1) # [N * text_num_per_video (T), C]
        t_feat = torch.cat(GatherLayer.apply(t_feat), dim=0)

        return self.head(v_s_feat, t_feat)

    def forward_test(self, imgs, subtitle_texts_item, texts_item):
        N = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        v_feat = self.encoder_v(imgs, N)

        for key in subtitle_texts_item:
            subtitle_texts_item[key] = subtitle_texts_item[key].reshape((-1,) + subtitle_texts_item[key].shape[2:])
        s_feat = self.encoder_t(subtitle_texts_item)

        v_s_feat = nn.functional.normalize(self.img_subtitle_mlp(torch.cat((v_feat, s_feat), dim=1)), dim=1)

        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        t_feat = nn.functional.normalize(self.encoder_t(texts_item), dim=1)  # [N * text_num_per_video (T), C]

        return zip(v_s_feat.cpu().numpy(),t_feat.view(N, -1, t_feat.shape[1]).cpu().numpy())

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
        subtitle_texts_item = data_batch['subtitle_texts_item']
        texts_item = data_batch['texts_item']
        losses, metric = self(imgs, subtitle_texts_item, texts_item)

        loss, log_vars = self._parse_losses(losses)

        for key, value in metric.items():
            if dist.is_available() and dist.is_initialized():
                value = value.data.clone()
                dist.all_reduce(value.div_(dist.get_world_size()))
            log_vars[key] = value.item()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        pass

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


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out