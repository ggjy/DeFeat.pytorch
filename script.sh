# VOC: train ResNet-50 based Faster RCNN 
python -m torch.distributed.launch --nproc_per_node=1 tools/train.py --gpus 1 --launcher pytorch --validate --work-dir /tmp/coco_output/kd/r50/ --config configs/faster_rcnn/voc_faster_rcnn_r50.py

# VOC: train ResNet-101 based Faster RCNN 
python -m torch.distributed.launch --nproc_per_node=1 tools/train.py --gpus 1 --launcher pytorch --validate --work-dir /tmp/coco_output/kd/r101/ --config configs/faster_rcnn/voc_faster_rcnn_r101.py

# KD FGFI: train student 
python -m torch.distributed.launch --nproc_per_node=1 tools/train_kd.py --gpus 1 --launcher pytorch --validate --work-dir /tmp/coco_output/kd/r101-50-FGFI/ --config configs/kd_faster_rcnn/voc_stu_faster_rcnn_r50_FGFI.py --config-t configs/kd_faster_rcnn/voc_tea_faster_rcnn_r101.py --checkpoint-t /tmp/coco_output/voc_bn_r101_8x2_lr0.02/epoch_4.pth

# KD DeFeat: train student 
To be added.