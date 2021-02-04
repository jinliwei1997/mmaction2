import torch.nn as nn
import torch
import torch.distributed as dist

from ..registry import HEADS

@HEADS.register_module
class ContrastiveHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self,
        temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.temperature = temperature


    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass


    def forward(self, x, y):
        """Forward head.

        Args:
            x (Tensor): [N * 256]
            y (Tensor): [N * text_num_per_video(T), 256]
            N : batch_size
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        # Similarity Matrix
        x = concat_all_gather(x)
        y = concat_all_gather(y)
        s = torch.matmul(x, y.permute(1, 0)) # (N) * (N * T)
        s = s.view(x.shape[0], x.shape[0], -1) # N * N * T

        # MIL-NCE loss
        nominator = s * torch.eye(s.shape[0])[:, :, None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((s, s.permute(1, 0, 2)), dim=1).view(s.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)

        losses = dict()
        losses['loss'] = torch.mean(denominator - nominator)

        return losses

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: dist.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output