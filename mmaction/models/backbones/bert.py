from transformers import BertTokenizer
from transformers import AutoModel
from transformers import pipeline
import torch.nn as nn
import torch

from ...utils import get_root_logger
from ..registry import BACKBONES


@BACKBONES.register_module()
class BERT(nn.Module):
    """BERT backbone.
    """
    def __init__(self,
                 pretrained = None,
                 freeze = True):
        super(BERT, self).__init__()
        self.pretrained = pretrained
        self.freeze = freeze

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

        if self.freeze:
            with torch.no_grad():
                text_out = self.model(**x).pooler_output
        else :
            text_out = self.model(**x).pooler_output
        return text_out
