# yolo-v2_pytorch
📙 yolo v2 re-implementation

### Introduction

This repository is about pytorch implementation of [YOLO9000](https://arxiv.org/abs/1612.08242). 

### Requirements

- Python 3.8
- Pytorch 1.7.1
- matplotlib
- visdom
- pillow

### Datasets

- [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)

```
  VOCDevkit
  ├──TRAIN
  │  ├── VOC2007
  │  │   ├── Annotations  
  │  │   ├── JPEGImages
  │  └── └── ...
  └──TEST
     ├── VOC2007
     │   ├── Annotations  
     │   ├── JPEGImages
     └── └── ...
	
```

- [COCO](https://cocodataset.org/#download)

```
COCO
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
└── images
	  ├── train2017
	  └── val2017
	
```

ref : https://github.com/csm-kr/yolo_v2_vgg16_pytorch
