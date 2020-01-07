# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .vgg import VGG16, PlainBlockBase, build_vgg_backbone
from .vgg_mrrp import build_mrrp_vgg_backbone
from .resnet_ws import build_ws_resnet_backbone, make_stage
from .resnet_ws_mrrp import build_mrrp_ws_resnet_backbone

# TODO can expose more resnet blocks after careful consideration
