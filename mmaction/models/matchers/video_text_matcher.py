from ..registry import MATCHERS
from .base import BaseMatcher
import torch.nn as nn
import torch
from .. import builder
from mmcv.cnn import normal_init
import torch.distributed as dist

@MATCHERS.register_module()
class VideoTextMatcher(BaseMatcher):
    """VideoTextMatcher model framework."""
    def __init__(self,
        backbone1,
        backbone2,
        head,
        queue_len=65536,
        neck=None,
        train_cfg=None,
        test_cfg=None,
        fp16_enabled=False,
        img_feat_dim = 2048,
        text_feat_dim = 768,
        feature_dim = 256,
        init_std = 0.01):
        super(VideoTextMatcher, self).__init__(backbone1,backbone2,head,train_cfg,test_cfg,fp16_enabled)

        self.img_feat_dim = img_feat_dim
        self.text_feat_dim = text_feat_dim
        self.feature_dim = feature_dim
        self.init_std = init_std

        self.img_mlp = nn.Sequential(nn.Linear(img_feat_dim, self.feature_dim * 2), nn.BatchNorm1d(self.feature_dim * 2), nn.ReLU(), nn.Linear(self.feature_dim * 2, self.feature_dim))
        self.text_mlp = nn.Sequential(nn.Linear(text_feat_dim, self.feature_dim * 2), nn.BatchNorm1d(self.feature_dim * 2), nn.ReLU(), nn.Linear(self.feature_dim * 2, self.feature_dim))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.init_mlp_weights()

        self.queue_len = queue_len

        # create v_queue and t_queue
        self.register_buffer("v_queue", torch.randn(feature_dim, queue_len))
        self.v_queue = nn.functional.normalize(self.v_queue, dim=0)
        self.register_buffer("v_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("t_queue", torch.randn(feature_dim, queue_len))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))



    def init_mlp_weights(self):
        """Initialize the model network weights."""
        for layer in self.img_mlp:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.text_mlp:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)

    def encoder_v(self, imgs, N):
        x = self.backbone1(imgs)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.reshape((N, -1) + x.shape[1:])
        x = x.mean(dim=1, keepdim=True)
        x = x.squeeze(1)
        # dropout
        x = x.view(x.size(0), -1)
        x = self.img_mlp(x)
        return x

    def encoder_t(self, texts):
        x = self.backbone2(texts)
        x = self.text_mlp(x)
        return x

    @torch.no_grad()
    def _dequeue_and_enqueue_v(self, keys):
        """Update v_queue."""
        # gather keys before updating v_queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.v_queue_ptr)
        if ptr + batch_size > self.queue_len:
            self.v_queue[:, ptr:] = keys.transpose(0, 1)[:, :self.queue_len-ptr]
            self.v_queue[:, :ptr + batch_size-self.queue_len] = keys.transpose(0, 1)[:, self.queue_len-ptr:]
            ptr = (ptr + batch_size) % self.queue_len  # move pointer
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.v_queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
            ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.v_queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_t(self, keys):
        """Update t_queue."""
        # gather keys before updating t_queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.t_queue_ptr)
        print(ptr)
        if ptr + batch_size > self.queue_len:
            self.t_queue[:, ptr:] = keys.transpose(0, 1)[:, :self.queue_len - ptr]
            self.t_queue[:, :ptr + batch_size - self.queue_len] = keys.transpose(0, 1)[:, self.queue_len - ptr:]
            ptr = (ptr + batch_size) % self.queue_len  # move pointer
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.t_queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
            ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.t_queue_ptr[0] = ptr

    def forward_train(self, imgs, texts_item):

        N = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        v_feat = nn.functional.normalize(self.encoder_v(imgs, N), dim=1) # [N , C]

        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])

        t_feat = nn.functional.normalize(self.encoder_t(texts_item), dim=1) # [N * text_num_per_video (T), C]

        if self.neck is not None:
            v_feat, t_feat = self.neck(v_feat, t_feat)

        # positive logits: [N , T]
        l_pos = torch.bmm(v_feat.view(N,1,self.feature_dim),t_feat.view(N,-1,self.feature_dim).transpose(1,2)).view(N,-1)

        # V->T negative logits: [N , t_queue_len]
        vt_l_neg = torch.mm(v_feat, self.t_queue.clone().detach())

        # # T->V negative logits: [N * T , v_queue_len]
        # tv_l_neg = torch.mm(t_feat, self.v_queue.clone().detach())

        losses, recall = self.head(l_pos, vt_l_neg)

        # self._dequeue_and_enqueue_v(v_feat)
        self._dequeue_and_enqueue_t(t_feat)


        return losses, recall

    def forward_test(self, imgs, texts_item):
        pass

    def forward_gradcam(self, audios):
        raise NotImplementedError

    def forward(self, imgs, texts_item, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, texts_item)

        return self.forward_test(imgs, texts_item)

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
        losses, recall = self(imgs, texts_item)

        loss, log_vars = self._parse_losses(losses)

        for key, value in recall.items():
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
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        pass

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output