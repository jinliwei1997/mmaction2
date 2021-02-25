import torch.nn as nn
import torch
import torch.distributed as dist

from ..registry import HEADS

@HEADS.register_module
class MILNCEHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self,
        temperature=0.1):
        super(MILNCEHead, self).__init__()
        self.temperature = temperature


    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass


    def forward(self, v_feat, t_feat):
        """Forward head.

        Args:
            v_feat (Tensor): [N * 256]
            t_feat (Tensor): [N * text_num_per_video(T), 256]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        v_feat = torch.true_divide(v_feat, self.temperature)
        t_feat = torch.true_divide(t_feat, self.temperature)
        s = torch.matmul(v_feat, t_feat.permute(1, 0)) # [N , N * T]
        s = s.view(v_feat.shape[0], v_feat.shape[0], -1) # [N , N , T]

        # MIL-NCE loss
        nominator = s * torch.eye(s.shape[0])[:, :, None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((s, s.permute(1, 0, 2)), dim=1).view(s.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)


        losses = dict()
        losses['mil_nce_loss'] = torch.mean(denominator - nominator)

        with torch.no_grad():
            N = s.shape[0]
            T = s.shape[2]
            s = s.view(v_feat.shape[0], -1)  # [N , N * T]

            _,  rank = torch.sort(s, dim=1, descending=True)

            recall1 = torch.zeros(N)
            recall5 = torch.zeros(N)
            recall10 = torch.zeros(N)
            avg_rank = torch.zeros(N)
            for i in range(N):
                for j in range(N*T):
                    if rank[i][j].item() >= T * i and rank[i][j].item() < T * (i + 1):
                        avg_rank[i] += j
                        if j<10:
                            recall10 += 1
                        if j<5:
                            recall5 += 1
                        if j<1:
                            recall1 += 1

            recall1 = torch.true_divide(recall1, T)
            recall5 = torch.true_divide(recall5, T)
            recall10 = torch.true_divide(recall10, T)
            avg_rank = torch.true_divide(avg_rank, T)

            meta = dict()
            meta['recall1'] = torch.mean(recall1)
            meta['recall5'] = torch.mean(recall5)
            meta['recall10'] = torch.mean(recall10)
            meta['avg_rank'] = torch.mean(avg_rank)

        """
        # tv_loss
        tv_nominator = torch.logsumexp(l_pos.view(-1, 1), dim=1) # [N * T]
        tv_logits = torch.cat((l_pos.view(-1, 1), tv_l_neg), dim=1) # [N * T, 1 + v_queue_len]
        tv_denominator = torch.logsumexp(tv_logits, dim=1)

        losses['tv_loss'] = torch.mean(tv_denominator - tv_nominator)
        """

        return losses, meta

