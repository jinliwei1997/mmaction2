from ..registry import MATCHERS
from .base import BaseMatcher
import torch.nn as nn
from .. import builder
from mmcv.cnn import normal_init
import torch.distributed as dist
import torch

@MATCHERS.register_module()
class VideoAudioTextMatcherE2E(nn.Module):
    """VideoTextMatcher model framework."""
    def __init__(self,
        v_backbone,
        a_backbone,
        t_backbone,
        head,
        neck = None,
        train_cfg = None,
        test_cfg = None,
        img_feat_dim = 2048,
        audio_feat_dim = 2048,
        text_feat_dim = 768,
        feature_dim = 256,
        init_std = 0.01,
        use_text_mlp = True):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.v_backbone = builder.build_backbone(v_backbone)
        self.a_backbone = builder.build_backbone(a_backbone)
        self.t_backbone = builder.build_backbone(t_backbone)
        self.head = builder.build_head(head)
        self.img_feat_dim = img_feat_dim
        self.audio_feat_dim = audio_feat_dim
        self.text_feat_dim = text_feat_dim
        self.feature_dim = feature_dim
        self.init_std = init_std

        self.img_audio_mlp = nn.Sequential(nn.Linear(img_feat_dim + audio_feat_dim, self.feature_dim * 2), nn.BatchNorm1d(self.feature_dim * 2), nn.ReLU(), nn.Linear(self.feature_dim * 2, self.feature_dim))
        self.text_mlp = nn.Sequential(nn.Linear(text_feat_dim, self.feature_dim * 2), nn.BatchNorm1d(self.feature_dim * 2), nn.ReLU(), nn.Linear(self.feature_dim * 2, self.feature_dim))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.use_text_mlp = use_text_mlp
        self.init_weights()

    def init_weights(self):
        """Initialize the model network weights."""
        self.v_backbone.init_weights()
        self.a_backbone.init_weights()
        self.t_backbone.init_weights()
        self.head.init_weights()
        self.init_mlp_weights()

    def init_mlp_weights(self):
        """Initialize the model network weights."""
        for layer in self.img_audio_mlp:
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

    def encoder_a(self, audios, N):
        x = self.a_backbone(audios)
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

    def forward(self, imgs, audios, texts_item, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, audios, texts_item)

        return self.forward_test(imgs, audios, texts_item)

    def forward_train(self, imgs, audios, texts_item):

        N = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        v_feat = self.encoder_v(imgs, N)

        audios = audios.reshape((-1,) + audios.shape[2:])
        a_feat = self.encoder_a(audios,N)
        v_a_feat = nn.functional.normalize(self.init_mlp_weights(torch.cat((v_feat, a_feat), dim=1),dim=1))
        v_a_feat = torch.cat(GatherLayer.apply(v_a_feat), dim=0)
        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        t_feat = nn.functional.normalize(self.encoder_t(texts_item), dim=1) # [N * text_num_per_video (T), C]
        t_feat = torch.cat(GatherLayer.apply(t_feat), dim=0)

        return self.head(v_a_feat, t_feat)

    def forward_test(self, imgs, audios, texts_item):
        N = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        v_feat = self.encoder_v(imgs, N)

        audios = audios.reshape((-1,) + audios.shape[2:])
        a_feat = self.encoder_a(audios, N)
        v_a_feat = nn.functional.normalize(self.init_mlp_weights(torch.cat((v_feat, a_feat), dim=1), dim=1))
        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        t_feat = nn.functional.normalize(self.encoder_t(texts_item), dim=1)  # [N * text_num_per_video (T), C]
        t_feat = torch.cat(GatherLayer.apply(t_feat), dim=0)

        return zip(v_a_feat.cpu().numpy(),t_feat.view(N, -1, v_feat.shape[1]).cpu().numpy())

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
        audios = data_batch['audios']
        texts_item = data_batch['texts_item']
        losses, metric = self(imgs, audios, texts_item)

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