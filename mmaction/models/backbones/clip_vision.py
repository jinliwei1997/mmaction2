import torch.nn as nn
import torch

from ...utils import get_root_logger
from ..registry import BACKBONES

import clip


@BACKBONES.register_module()
class CLIPViT(nn.Module):
    def __init__(self, pretrained=None, freeze=True, fp16_enabled=True):
        super().__init__()
        self.pretrained = pretrained
        self.freeze = freeze
        self.fp16_enabled = fp16_enabled

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            assert (
                    self.pretrained in clip.available_models()
            ), "not allowed pretrained model"
            logger = get_root_logger()
            logger.info(f"load model from: {self.pretrained}")
            self.model = clip.load(self.pretrained)[0].visual
        else:
            raise TypeError("pretrained must be a str")

    def forward(self, x):
        # x.shape = [batch * seg, C, H, W]
        if self.fp16_enabled:
            x = x.half()
        if self.freeze:
            self.model.eval()
            with torch.no_grad():
                # features = self.model(x)
                x = self.model.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(
                    x.shape[0], x.shape[1], -1
                )  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat(
                    [
                        self.model.class_embedding.to(x.dtype)
                        + torch.zeros(
                            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                        ),
                        x,
                        ],
                    dim=1,
                )  # shape = [*, grid ** 2 + 1, width]
                x = x + self.model.positional_embedding.to(x.dtype)
                x = self.model.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.model.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD

                features = self.model.ln_post(x[:, 0, :])
        else:
            # features = self.model(x)
            x = self.model.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    self.model.class_embedding.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                    ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            x = x + self.model.positional_embedding.to(x.dtype)
            x = self.model.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.model.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            features = self.model.ln_post(x[:, 0, :])
        # x.shape = [batch * seg, 768]
        return features