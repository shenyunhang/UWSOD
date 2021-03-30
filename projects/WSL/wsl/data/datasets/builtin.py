# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.register_coco import register_coco_instances

from .builtin_meta import _get_builtin_metadata

# fmt: off
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# fmt: on

# ==== Predefined datasets and splits for Flickr ==========

_PREDEFINED_SPLITS_WEB = {}
_PREDEFINED_SPLITS_WEB["flickr"] = {
    "flickr_voc": ("flickr_voc/images", "flickr_voc/images_d2.json"),
    "flickr_coco": ("flickr_coco/images", "flickr_coco/images_d2.json"),
}


def register_all_web(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_WEB.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(key),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined datasets and splits for VOC_PGT ==========

_PREDEFINED_SPLITS_VOC_PGT = {}
_PREDEFINED_SPLITS_VOC_PGT["voc_2007_pgt"] = {
    "voc_2007_train_pgt": (
        "VOC2007/JPEGImages",
        "VOC2007/../results/VOC2007/Main/voc_2007_train_pgt.json",
    ),
    "voc_2007_val_pgt": (
        "VOC2007/JPEGImages",
        "VOC2007/../results/VOC2007/Main/voc_2007_val_pgt.json",
    ),
}


def register_all_voc_pgt(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VOC_PGT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(key),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# Register them all under "./datasets"
_root = os.getenv("wsl_DATASETS", "datasets")
register_all_web(_root)
register_all_voc_pgt(_root)
