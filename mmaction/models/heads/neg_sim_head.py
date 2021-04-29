import torch.nn as nn
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ..registry import HEADS
import numpy as np
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
            s1 = torch.matmul(v_feat, p_t.permute(1, 0)).view(N,N)
            s2 = torch.matmul(t_feat, p_v.permute(1, 0)).view(N,N)

            v_metric = self.retrieval_metric(s1.cpu())
            metric['v_recall1'] = v_metric['R1'].cuda()
            metric['v_recall5'] = v_metric['R5'].cuda()
            metric['v_recall10'] = v_metric['R10'].cuda()
            metric['v_med_rk'] = v_metric['MR'].cuda()

            t_metric = self.retrieval_metric(s2.cpu())
            metric['t_recall1'] = v_metric['R1'].cuda()
            metric['t_recall5'] = v_metric['R5'].cuda()
            metric['t_recall10'] = v_metric['R10'].cuda()
            metric['t_med_rk'] = v_metric['MR'].cuda()

        return losses, metric

    def retrieval_metric(self, x):
        sx = np.sort(-x, axis=1)
        d = np.diag(-x)
        d = d[:, np.newaxis]
        ind = sx - d
        ind = np.where(ind == 0)
        ind = ind[1]
        metrics = {}
        metrics['R1'] = torch.Tensor(float(np.sum(ind == 0)) / len(ind))
        metrics['R5'] = torch.Tensor(float(np.sum(ind < 5)) / len(ind))
        metrics['R10'] = torch.Tensor(float(np.sum(ind < 10)) / len(ind))
        metrics['MR'] = torch.Tensor(np.median(ind) + 1)

        return metrics