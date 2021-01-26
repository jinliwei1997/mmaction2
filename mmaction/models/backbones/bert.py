from transformers import BertTokenizer
from transformers import AutoModel
from transformers import pipeline
import torch.nn as nn

from ...utils import get_root_logger
from ..registry import BACKBONES


@BACKBONES.register_module()
class BERT(nn.Module):
    """BERT backbone.
    """
    def __init__(self,
                 pretrained=None):
        super.__init__()
        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            self.model = AutoModel.from_pretrained(self.pretrained).to('cuda')
            self.model.train()
        else:
            raise TypeError('pretrained must be a str')

    def forward(self, x):
        return self.model(x)
