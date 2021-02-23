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


    def forward(self, l_pos, vt_l_neg, tv_l_neg):
        """Forward head.

        Args:
            l_pos: positive logits: [N , T]
            vt_l_neg: V->T negative logits: [N , t_queue_len]
            tv_l_neg: T->V negative logits: [N * T , v_queue_len]

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        l_pos /= self.temperature
        vt_l_neg /= self.temperature
        tv_l_neg /= self.temperature

        losses = dict()


        # vt_loss
        vt_nominator = torch.logsumexp(l_pos, dim=1)
        vt_logits = torch.cat((l_pos, vt_l_neg), dim=1) # [N, T + t_queue_len]
        vt_denominator = torch.logsumexp(vt_logits, dim=1)

        losses['vt_loss'] = torch.mean(vt_denominator - vt_nominator)

        """
        # tv_loss
        tv_nominator = torch.logsumexp(l_pos.view(-1, 1), dim=1) # [N * T]
        tv_logits = torch.cat((l_pos.view(-1, 1), tv_l_neg), dim=1) # [N * T, 1 + v_queue_len]
        tv_denominator = torch.logsumexp(tv_logits, dim=1)

        losses['tv_loss'] = torch.mean(tv_denominator - tv_nominator)
        """

        return losses

