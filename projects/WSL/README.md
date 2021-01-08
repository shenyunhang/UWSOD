# UWSOD: Toward Fully-Supervised-Level Capacity Weakly Supervised Object Detection

By [Yunhang Shen](), [Rongrong Ji](), [Zhiwei Chen](), [Yongjian Wu](), [Feiyue Huang]().

NeurIPS 2020 Paper.

This project is based on [Detectron2](https://github.com/facebookresearch/detectron2).

## License

UWSOD is released under the [Apache 2.0 license](LICENSE).


## Citing UWSOD

If you find UWSOD useful in your research, please consider citing:

```
@inproceedings{UWSOD_2020_NeurIPS,
	author = {Shen, Yunhang and Ji, Rongrong and Chen, Zhiwei and Wu, Yongjian and Huang, Feiyue},
	title = {UWSOD: Toward Fully-Supervised-Level Capacity Weakly Supervised Object Detection},
	booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
	year = {2020},
}   
```

## Installation

Install our forked Detectron2:
```
git clone https://github.com/shenyunhang/UWSOD.git
cd UWSOD
python3 -m pip install -e .
```
If you have problem of installing Detectron2, please checking [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

Install UWSOD project:
```
cd projects/WSL
pip3 install -r requirements.txt
git submodule update --init --recursive
python3 -m pip install -e .
cd ../../
```

## Dataset Preparation
Please follow [this](https://github.com/shenyunhang/UWSOD/blob/UWSOD/datasets/README.md#expected-dataset-structure-for-pascal-voc) to creating symlinks for PASCAL VOC.


## Model Preparation

Download models from this [here](https://1drv.ms/f/s!Am1oWgo9554dgRQ8RE1SRGvK7HW2):
```
mv models $UWSOD
```

Then we have the following directory structure:
```
UWSOD
|_ models
|  |_ DRN-WSOD
|     |_ resnet18_ws_model_120_d2.pkl
|     |_ resnet150_ws_model_120_d2.pkl
|     |_ resnet101_ws_model_120_d2.pkl
|_ ...
```


## Quick Start: Using UWSOD

### UWSOD

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/uwsod_WSR_18_DC5_1x.yaml OUTPUT_DIR output/uwsod_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

We also provide our training log and final model [here](https://1drv.ms/u/s!Am1oWgo9554dhuBqu66fRrzkl8wBzg?e=dlKN4W).

To run inference with test-time augmentation on this final model:
```
python3 projects/WSL/tools/train_net.py --resume --eval-only --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/uwsod_WSR_18_DC5_1x.yaml OUTPUT_DIR output/uwsod_WSR_18_DC5_VOC07_2020-12-02_12-59-45
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/uwsod_V_16_DC5_1x.yaml OUTPUT_DIR output/uwsod_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```
