# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn

from .utils import load_checkpoint
from .utils import match_layers_and_load_checkpoint


BACKBONES_FOR_LAYER_MATCHING = ['regnet_x_800mf',
                                'resnet18'
                                ]


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def init_weights(self, pretrained=None):
        """Init backbone weights.

        Args:
            pretrained (str | None): If pretrained is a string, then it
                initializes backbone weights by loading the pretrained
                checkpoint. If pretrained is None, then it follows default
                initializer or customized initializer in subclasses.
        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            backbone_name = pretrained.split('//')[-1]
            if backbone_name not in BACKBONES_FOR_LAYER_MATCHING:
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                print(f"Trying to match state_dict and model layers for {backbone_name}")
                match_layers_and_load_checkpoint(self, pretrained, logger=logger)
        elif pretrained is None:
            # use default initializer or customized initializer in subclasses
            pass
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')

    @abstractmethod
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
