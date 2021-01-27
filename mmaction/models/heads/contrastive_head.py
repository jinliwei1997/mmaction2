import torch.nn as nn
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
        self.criterion = nn.CrossEntropyLoss()


    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.img_fc, std=self.init_std)
        normal_init(self.text_fc, std=self.init_std)

    def forward(self, x, y, N):
        """Forward head.

        Args:
            x (Tensor): [N * num_segs, img_in_channels, 7, 7]
            y (Tensor): [N * num_per_video, text_in_channels]
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
        print('x_hidden_shape: ', x_hidden.shape)
        y_hidden = self.text_fc(y)
        print('y_hidden_shape: ', y_hidden.shape)

        # N = pos.size(0)
        # logits = torch.cat((pos, neg), dim=1)
        # logits /= self.temperature
        # labels = torch.zeros((N, ), dtype=torch.long).cuda()
        #
        # losses = dict()
        # losses['loss'] = self.criterion(logits, labels)
        return x