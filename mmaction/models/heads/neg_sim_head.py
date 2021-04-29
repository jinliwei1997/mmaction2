import torch.nn as nn
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ..registry import HEADS

@HEADS.register_module
class NegSimHead(nn.Module):
    """Head for RankingLoss.

    """
    def __init__(self):
        super(NegSimHead, self).__init__()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, v_feat, t_feat, p_v, p_t):
        """Forward head.

        Args:
            v_feat (Tensor): [N , C]
            t_feat (Tensor): [N , C]
            p_v (Tensor): [N , C]
            p_t (Tensor): [N , C]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        v_feat = v_feat.detach()
        t_feat = t_feat.detach()
        v_feat = F.normalize(v_feat, dim=1)
        t_feat = F.normalize(t_feat, dim=1)
        p_v = F.normalize(p_v, dim=1)
        p_t = F.normalize(p_t, dim=1)

        losses = dict()
        losses['neg_sim_loss'] = -0.5*(p_v * t_feat).sum(dim=1).mean() -0.5*(p_t * v_feat).sum(dim=1).mean()


        with torch.no_grad():

            metric = {}
            metric['v_feat_std'] = torch.mean(torch.std(v_feat, dim = 0))
            metric['t_feat_std'] = torch.mean(torch.std(t_feat, dim = 0))
            metric['p_v_std'] = torch.mean(torch.std(p_v, dim = 0))
            metric['p_t_std'] = torch.mean(torch.std(p_t, dim = 0))

            N = v_feat.shape[0]
            s1 = torch.matmul(v_feat, p_t.permute(1, 0)).view(N,N,-1)
            s2 = torch.matmul(t_feat, p_v.permute(1, 0)).view(N,N,-1)

            recall1, recall5, recall10, mean_rk = self.retrieval_metric(s1)
            metric['v_recall1'] = recall1
            metric['v_recall5'] = recall5
            metric['v_recall10'] = recall10
            metric['v_mean_rk'] = mean_rk

            recall1, recall5, recall10, mean_rk = self.retrieval_metric(s2)
            metric['t_recall1'] = recall1
            metric['t_recall5'] = recall5
            metric['t_recall10'] = recall10
            metric['t_mean_rk'] = mean_rk

        return losses, metric

    def retrieval_metric(self, s):
        with torch.no_grad():
            N = s.shape[0]
            T = s.shape[2]
            s = s.view(N, -1)  # [N , N * T]

            _, rank = torch.sort(s, dim=1, descending=True)

            recall1 = torch.zeros(N).cuda()
            recall5 = torch.zeros(N).cuda()
            recall10 = torch.zeros(N).cuda()
            mean_rk = torch.zeros(N).cuda()
            for i in range(N):
                for j in range(N * T):
                    if rank[i][j].item() >= T * i and rank[i][j].item() < T * (i + 1):
                        mean_rk[i] += j
                    if j < 10:
                        recall10[i] += 1
                    if j < 5:
                        recall5[i] += 1
                    if j < 1:
                        recall1[i] += 1

            recall1 = torch.true_divide(recall1, T)
            recall5 = torch.true_divide(recall5, T)
            recall10 = torch.true_divide(recall10, T)
            mean_rk = torch.true_divide(mean_rk, T)

        return recall1, recall5, recall10, mean_rk