# DeFeat.pytorch Code Base

Implementation of our CVPR2021 paper [Distilling Object Detectors via Decoupled Features](https://arxiv.org/pdf/2103.14475.pdf)

### Abstract

Knowledge distillation is a widely used paradigm for inheriting information from a complicated teacher network to a compact student network and maintaining the strong performance. Different from image classification, object detectors are much more sophisticated with multiple loss functions in which features that semantic information rely on are tangled. In this paper, we point out that the information of features derived from regions excluding objects are also essential for distilling the student detector, which is usually ignored in existing approaches. In addition, we elucidate that features from different regions should be assigned with different importance during distillation. To this end, we present a novel distillation algorithm via decoupled features (DeFeat) for learning a better student detector. Specifically, two levels of decoupled features will be processed for embedding useful information into the student, i.e., decoupled features from neck and decoupled proposals from classification head. Extensive experiments on various detectors with different backbones show that the proposed DeFeat is able to surpass the state-of-the-art distillation methods for object detection. For example, DeFeat improves ResNet50 based Faster R-CNN from 37.4% to 40.9% mAP, and improves ResNet50 based RetinaNet from 36.5% to 39.7% mAP on COCO benchmark. 

### Environments
- Python 3.7
- MMDetection 2.x
- This repo uses: `mmdet-v2.0` `mmcv-0.5.6` `cuda 10.1`

### VOC Results

**Notes:**

- Faster RCNN based model
- Batch: sample_per_gpu x gpu_num


| Model | BN | Grad clip | Batch | Lr schd | box AP | Model | Log |
|:-----:|:--:|:---------:|:-----:|:-------:|:------:|:-----:|:---:|
| R101  | bn | None      |  8x2  | 0.01    | 81.70  | |     |
| R101  | bn | None      |  8x2  | 0.02    | 82.27  | | [GoogleDrive](https://drive.google.com/file/d/1KqmlLZMWxa264Z-PjFLD08iw_lmiFyDK/view?usp=sharing) |
| R101  | syncbn | max=35 | 8x2  | 0.01    | 81.59  | |     |
| R101  | syncbn | None  |  8x2  | 0.02    | 81.83  | |     |
| R50   | bn | max=35    |  8x2  | 0.02    | 80.97  | |     |
| R50   | syncbn | None  |  8x2  | 0.02    | 80.76  | |     |
| R50   | syncbn | max=35 | 8x2  | 0.01    | 80.66  | |     |
| R50   | bn | None      |  8x2  | 0.01    | 80.52  | | [GoogleDrive](https://drive.google.com/file/d/16-trLtFphZQegdf0aB9QCndo2m8Owndg/view?usp=sharing) |
| R101-50-FGFI-w1 | bn | max=35 | 8x2  | 0.01    | 81.04 | |  [GoogleDrive](https://drive.google.com/file/d/1xUjBUrx54-r6byDH0vs-YkNsz2l0F8ZJ/view?usp=sharing)   |
| R101-50-FGFI-w2 | bn | max=35 | 8x2  | 0.01    | 81.17 | | [GoogleDrive](https://drive.google.com/file/d/1YKsDP87zNIJp9ucITL6v_T1OpuHbWbuA/view?usp=sharing)    |
| R101-50-FGFI-w2 | bn | max=35 | 8x2  | 0.01    | 82.04 | | [GoogleDrive](https://drive.google.com/file/d/1_80QQVhXgjydHjZvMCt2pdKQaL9sW-rC/view?usp=sharing)    |


## Acknowledgement
Our code is based on the open source project [MMDetection](https://github.com/open-mmlab/mmdetection).
