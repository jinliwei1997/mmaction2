from ..registry import MATCHERS
from .base import BaseMatcher


@MATCHERS.register_module()
class VideoTextMatcher(BaseMatcher):
    """Audio recognizer model framework."""

    def forward(self, imgs, texts, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, texts)

        return self.forward_test(imgs, texts)

    def forward_train(self, imgs, texts):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.backbone1(imgs)
        print('x shape:', x.shape)
        print('num_segs', num_segs)
        for key in texts:
            texts[key] = texts[key].reshape((-1,) + texts[key].shape[2:])
        print(texts)
        y = self.backbone2(texts)
        print('y:', y)
        print('y shape:', y.shape)
        if self.neck is not None:
            x,y = self.neck(x,y)

        loss = self.head.loss(x,y)

        return loss

    def forward_test(self, imgs, texts):
        """Defines the computation performed at every call when training."""
        x = self.backbone1(imgs)
        y = self.backbone2(texts)
        if self.neck is not None:
            x,y = self.neck(x, y)
        loss = self.head.loss(x, y)

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
        texts = data_batch['texts_item']
        losses = self(imgs, texts)

        loss, log_vars = self._parse_losses(losses)

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
        texts = data_batch['texts_item']

        losses = self(imgs, texts)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
