# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_wsl_config(cfg):
    """
    Add config for mrrpnet.
    """
    _C = cfg

    _C.MODEL.VGG = CN()

    _C.MODEL.VGG.DEPTH = 16
    _C.MODEL.VGG.OUT_FEATURES = ["plain5"]
    _C.MODEL.VGG.CONV5_DILATION = 1

    _C.WSL = CN()
    _C.WSL.VIS_TEST = False
    _C.WSL.ITER_SIZE = 1
    _C.WSL.MEAN_LOSS = True

    # CMIL
    _C.WSL.SIZE_EPOCH = 5000
    _C.WSL.CMIL = False

    _C.MODEL.ROI_BOX_HEAD.DAN_DIM = [4096, 4096]

    _C.WSL.USE_OBN = True

    _C.WSL.CSC_MAX_ITER = 35000

    _C.WSL.REFINE_NUM = 3
    _C.WSL.REFINE_REG = [False, False, False]
    _C.WSL.HAS_GAM = False
    _C.WSL.REFINE_MIST = False

    # List of the dataset names for validation. Must be registered in DatasetCatalog
    _C.DATASETS.VAL = ()
    # List of the pre-computed proposal files for val, which must be consistent
    # with datasets listed in DATASETS.VAL.
    _C.DATASETS.PROPOSAL_FILES_VAL = ()

    _C.MODEL.SEM_SEG_HEAD.ASSP_CONVS_DIM = [1024, 1024]
    _C.MODEL.SEM_SEG_HEAD.MASK_SOFTMAX = False
    _C.MODEL.SEM_SEG_HEAD.CONSTRAINT = False

    _C.TEST.EVAL_TRAIN = True

    _C.WSL.CLS_AGNOSTIC_BBOX_KNOWN = False

    _C.WSL.SAMPLING = CN()
    _C.WSL.SAMPLING.SAMPLING_ON = False
    _C.WSL.SAMPLING.IOU_THRESHOLDS = [[0.5], [0.5], [0.5], [0.5]]
    _C.WSL.SAMPLING.IOU_LABELS = [[0, 1], [0, 1], [0, 1], [0, 1]]
    _C.WSL.SAMPLING.BATCH_SIZE_PER_IMAGE = [4096, 4096, 4096, 4096]
    _C.WSL.SAMPLING.POSITIVE_FRACTION = [1.0, 1.0, 1.0, 1.0]

    _C.WSL.CASCADE_ON = False

    _C.MODEL.MRRP = CN()
    _C.MODEL.MRRP.MRRP_ON = False
    _C.MODEL.MRRP.NUM_BRANCH = 3
    _C.MODEL.MRRP.BRANCH_DILATIONS = [1, 2, 3]
    _C.MODEL.MRRP.MRRP_STAGE = "res4"
    _C.MODEL.MRRP.TEST_BRANCH_IDX = 1
