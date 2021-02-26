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

    def forward(self, v_feat, t_feat, return_loss=True):
        if return_loss:
            return self.forward_train(v_feat, t_feat)

        return self.forward_test(v_feat, t_feat)

    def forward_train(self, v_feat, t_feat):
        """Forward head.

        Args:
            v_feat (Tensor): [N * 256]
            t_feat (Tensor): [N * text_num_per_video(T), 256]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        s = torch.matmul(v_feat, t_feat.permute(1, 0)) # [N , N * T]
        s = torch.true_divide(s, self.temperature)
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

            recall1 = torch.zeros(N).cuda()
            recall5 = torch.zeros(N).cuda()
            recall10 = torch.zeros(N).cuda()
            avg_rank = torch.zeros(N).cuda()
            for i in range(N):
                for j in range(N*T):
                    if rank[i][j].item() >= T * i and rank[i][j].item() < T * (i + 1):
                        avg_rank[i] += j
                        if j<10:
                            recall10[i] += 1
                        if j<5:
                            recall5[i] += 1
                        if j<1:
                            recall1[i] += 1

            recall1 = torch.true_divide(recall1, T)
            recall5 = torch.true_divide(recall5, T)
            recall10 = torch.true_divide(recall10, T)
            avg_rank = torch.true_divide(avg_rank, T)

            metric = dict()
            metric['recall1'] = torch.mean(recall1)
            metric['recall5'] = torch.mean(recall5)
            metric['recall10'] = torch.mean(recall10)
            metric['avg_rank'] = torch.mean(avg_rank)

        """
        # tv_loss
        tv_nominator = torch.logsumexp(l_pos.view(-1, 1), dim=1) # [N * T]
        tv_logits = torch.cat((l_pos.view(-1, 1), tv_l_neg), dim=1) # [N * T, 1 + v_queue_len]
        tv_denominator = torch.logsumexp(tv_logits, dim=1)

        losses['tv_loss'] = torch.mean(tv_denominator - tv_nominator)
        """

        return losses, metric

    def forward_test(self, v_feat, t_feat):

        s = torch.matmul(v_feat, t_feat.permute(1, 0))  # [N , N * T]
        s = torch.true_divide(s, self.temperature)
        s = s.view(v_feat.shape[0], v_feat.shape[0], -1)  # [N , N , T]

        with torch.no_grad():
            N = s.shape[0]
            T = s.shape[2]
            s = s.view(v_feat.shape[0], -1)  # [N , N * T]

            _, rank = torch.sort(s, dim=1, descending=True)

            recall1 = torch.zeros(N).cuda()
            recall5 = torch.zeros(N).cuda()
            recall10 = torch.zeros(N).cuda()
            avg_rank = torch.zeros(N).cuda()

            recall1 = torch.true_divide(recall1, T)
            recall5 = torch.true_divide(recall5, T)
            recall10 = torch.true_divide(recall10, T)
            avg_rank = torch.true_divide(avg_rank, T)

            metric = torch.cat([recall1.view(N,1), recall5.view(N,1), recall10.view(N,1), avg_rank.view(N,1)], dim=1)
            
        return metric.cpu().numpy()