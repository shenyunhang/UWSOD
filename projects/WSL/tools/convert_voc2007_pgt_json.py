# -*- coding: utf-8 -*-
import argparse
import json
import os
import cv2
from tqdm import tqdm

root_dir_voc2007 = "datasets/VOC2007/"
image_dir_voc2007 = os.path.join(root_dir_voc2007, "JPEGImages/")
txt_dir_voc2007 = os.path.join(root_dir_voc2007, "ImageSets/Main")

anno_dir = os.path.join(root_dir_voc2007, "..", "results/VOC2007/Main/")
anno_ids = {"train": "", "val": ""}


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--train", help="predicted txt file pattern, i.e., comp3_xxxxx_det_train", required=True
    )
    parser.add_argument(
        "--val", help="predicted txt file pattern, i.e., comp3_xxxxx_det_val", required=True
    )

    return parser.parse_args()


categories_list = [
    {"supercategory": "none", "id": 1, "name": "aeroplane"},
    {"supercategory": "none", "id": 2, "name": "bicycle"},
    {"supercategory": "none", "id": 3, "name": "bird"},
    {"supercategory": "none", "id": 4, "name": "boat"},
    {"supercategory": "none", "id": 5, "name": "bottle"},
    {"supercategory": "none", "id": 6, "name": "bus"},
    {"supercategory": "none", "id": 7, "name": "car"},
    {"supercategory": "none", "id": 8, "name": "cat"},
    {"supercategory": "none", "id": 9, "name": "chair"},
    {"supercategory": "none", "id": 10, "name": "cow"},
    {"supercategory": "none", "id": 11, "name": "diningtable"},
    {"supercategory": "none", "id": 12, "name": "dog"},
    {"supercategory": "none", "id": 13, "name": "horse"},
    {"supercategory": "none", "id": 14, "name": "motorbike"},
    {"supercategory": "none", "id": 15, "name": "person"},
    {"supercategory": "none", "id": 16, "name": "pottedplant"},
    {"supercategory": "none", "id": 17, "name": "sheep"},
    {"supercategory": "none", "id": 18, "name": "sofa"},
    {"supercategory": "none", "id": 19, "name": "train"},
    {"supercategory": "none", "id": 20, "name": "tvmonitor"},
]


def read_txt(path, split="train"):
    txt_path = os.path.join(path, "{}.txt".format(split))
    with open(txt_path) as f:
        ids = f.readlines()
    return ids


def generate_anno(anno_dir, anno_id, split):
    anno_path_tmp = os.path.join(anno_dir, anno_id + "_{}.txt")
    anno_cls_path_tmp = os.path.join(txt_dir_voc2007, "{}_{}.txt")

    count = 0
    annotations = []
    for category in tqdm(categories_list):
        anno_path = anno_path_tmp.format(category["name"])
        anno_cls_path = anno_cls_path_tmp.format(category["name"], split)

        with open(anno_cls_path) as f:
            lines = f.readlines()
            pos_id = []
            for line in lines:
                line = line.strip()
                line = line.split()
                img_id = line[0]
                label = line[1]

                if label == "1":
                    pos_id.append(img_id)

        with open(anno_path) as f:
            lines = f.readlines()

            used_id = []
            for line in lines:
                line = line.strip()
                line = line.split()
                img_id = line[0]
                # score = line[1]
                x1 = float(line[2])
                y1 = float(line[3])
                x2 = float(line[4])
                y2 = float(line[5])

                if img_id not in pos_id:
                    continue
                if img_id in used_id:
                    continue
                used_id.append(img_id)

                w = x2 - x1
                h = y2 - y1
                area = int(w * h)

                anno = {
                    "area": area,
                    "image_id": img_id,
                    "bbox": [int(x1), int(y1), int(w), int(h)],
                    "iscrowd": 0,
                    "category_id": category["id"],
                    "id": count,
                }
                count += 1

                annotations.append(anno)
    return annotations


def generate_image_info(img_path, images_info):
    img = cv2.imread(img_path)

    img_name = img_path.split("/")[-1]

    img_w = img.shape[1]
    img_h = img.shape[0]

    info = {"file_name": img_name, "height": img_h, "width": img_w, "id": img_name[:-4]}
    images_info.append(info)

    return images_info


def save_json(ann, path, split="train"):
    os.system("mkdir -p {}".format(path))
    instance_path = os.path.join(path, "{}.json".format(split))
    with open(instance_path, "w") as f:
        json.dump(ann, f)


def convert_to_json(ids, split, save_split):
    images_info = []
    for i in tqdm(range(len(ids))):
        img_path = os.path.join(image_dir_voc2007, ids[i][:-1] + ".jpg")
        assert os.path.isfile(img_path)
        images_info = generate_image_info(img_path, images_info)

    annotations = generate_anno(anno_dir, anno_ids[split], split)

    voc_instance = {
        "images": images_info,
        "annotations": annotations,
        "categories": categories_list,
    }
    save_json(voc_instance, anno_dir, split=save_split)


def convert_voc2007():
    ids_train = read_txt(txt_dir_voc2007, "train")

    ids_val = read_txt(txt_dir_voc2007, "val")

    convert_to_json(ids_train, "train", "voc_2007_train_pgt")
    convert_to_json(ids_val, "val", "voc_2007_val_pgt")


if __name__ == "__main__":
    args = parse_arguments()
    anno_ids["train"] = args.train
    anno_ids["val"] = args.val
    convert_voc2007()
