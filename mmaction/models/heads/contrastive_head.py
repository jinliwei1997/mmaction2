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


    def forward(self, l_pos, vt_l_neg):
        """Forward head.

        Args:
            l_pos: positive logits: [N , T]
            vt_l_neg: V->T negative logits: [N , t_queue_len]
            # tv_l_neg: T->V negative logits: [N * T , v_queue_len]

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        T = l_pos.shape[1]
        l_pos /= self.temperature
        vt_l_neg /= self.temperature
        # tv_l_neg /= self.temperature
        print(f'l_pos shape: {l_pos.shape}')
        print(f'vt_l_neg shape: {vt_l_neg.shape}')
        losses = dict()
        recall = dict()

        # vt_loss
        vt_nominator = torch.logsumexp(l_pos, dim=1)
        vt_logits = torch.cat((l_pos, vt_l_neg), dim=1) # [N, T + t_queue_len]
        vt_denominator = torch.logsumexp(vt_logits, dim=1)

        losses['vt_loss'] = torch.mean(vt_denominator - vt_nominator)

        _, top1 = vt_logits.topk(k=1, dim=1)
        recall1 = torch.true_divide(torch.sum((top1 < T), dim=1), T)

        _, top5 = vt_logits.topk(k=5, dim=1)
        recall5 = torch.true_divide(torch.sum((top5 < T), dim=1), T)

        _, top10 = vt_logits.topk(k=10, dim=1)
        recall10 = torch.true_divide(torch.sum((top10 < T), dim=1), T)

        recall['recall1'] = torch.mean(recall1)
        recall['recall5'] = torch.mean(recall5)
        recall['recall10'] = torch.mean(recall10)

        """
        # tv_loss
        tv_nominator = torch.logsumexp(l_pos.view(-1, 1), dim=1) # [N * T]
        tv_logits = torch.cat((l_pos.view(-1, 1), tv_l_neg), dim=1) # [N * T, 1 + v_queue_len]
        tv_denominator = torch.logsumexp(tv_logits, dim=1)

        losses['tv_loss'] = torch.mean(tv_denominator - tv_nominator)
        """

        return losses, recall

