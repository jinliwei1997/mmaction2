import torch.nn as nn
import torch
from mmcv.cnn import normal_init

from ..registry import HEADS
from .base import AvgConsensus, BaseHead

@HEADS.register_module
class ContrastiveHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self,
        img_in_channels,
        text_in_channels,
        hidden_state_channels,
        init_std,
        consensus=dict(type='AvgConsensus', dim=1),
        temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.img_in_channels = img_in_channels
        self.text_in_channels = text_in_channels
        self.hidden_state_channels = hidden_state_channels
        self.init_std = init_std

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        self.temperature = temperature

        self.img_fc =  nn.Linear(self.img_in_channels, self.hidden_state_channels)
        self.text_fc = nn.Linear(self.text_in_channels, self.hidden_state_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.img_fc, std=self.init_std)
        normal_init(self.text_fc, std=self.init_std)

    def _create_buffer(N, T):

        pos_ind = (torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(1, T).view(-1, 1).squeeze().cuda(),
                  torch.arange(N * T).cuda())
        neg_mask = torch.ones((N, N * T), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0
        return pos_ind, neg_mask

    def forward(self, x, y, N):
        """Forward head.

        Args:
            x (Tensor): [N * num_segs, img_in_channels, 7, 7]
            y (Tensor): [N * text_num_per_video(T), text_in_channels]
            N : batch_size
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.reshape((N, -1) + x.shape[1:])
        x = self.consensus(x)
        x = x.squeeze(1)
        # dropout
        x = x.view(x.size(0), -1)
        x_hidden = self.img_fc(x)

        y_hidden = self.text_fc(y)

        # Similarity Matrix
        x_hidden = x_hidden / (torch.norm(x_hidden, p=2, dim=1, keepdim=True) + 1e-10)
        y_hidden = y_hidden / (torch.norm(y_hidden, p=2, dim=1, keepdim=True) + 1e-10)
        s = torch.matmul(x_hidden, y_hidden.permute(1, 0)) # (N) * (N * T)
        s = s.view(N, N, -1) # N * N * T

        # MIL-NCE loss
        nominator = s * torch.eye(s.shape[0])[:, :, None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((s, s.permute(1, 0, 2)), dim=1).view(s.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)

        losses = dict()
        losses['loss'] = torch.mean(denominator - nominator)

        return losses