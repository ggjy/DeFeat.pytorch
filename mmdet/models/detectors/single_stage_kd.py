import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class SingleStageDetectorKD(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 hint_adapt=dict(type='hint')):
        super(SingleStageDetectorKD, self).__init__()
        self.backbone = build_backbone(backbone)
        self.hint_adapt = hint_adapt
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if 'neck-adapt' in hint_adapt.type:
            self.neck_adapt = []
            in_channels = hint_adapt.neck_in_channels
            out_channels = hint_adapt.neck_out_channels
            for i in range(len(in_channels)):
                self.neck_adapt.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels[i], out_channels[i], kernel_size=3, padding=1),
                        # nn.ReLU(),
                        nn.Sequential()
                    )
                )
            self.neck_adapt = nn.ModuleList(self.neck_adapt)
        if 'bb-adapt' in hint_adapt.type:
            self.bb_adapt = []
            in_channels = hint_adapt.bb_in_channels
            out_channels = hint_adapt.bb_out_channels
            for i in range(len(in_channels)):
                self.bb_adapt.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels[i], out_channels[i], kernel_size=3, padding=1),
                        nn.ReLU()
                    )
                )
            self.bb_adapt = nn.ModuleList(self.bb_adapt)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetectorKD, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
        if 'neck-adapt' in self.hint_adapt.type:
            for i in range(len(self.neck_adapt)):
                self.neck_adapt[i][0].weight.data.normal_().fmod_(2).mul_(0.0001).add_(0)
                self.neck_adapt[i].cuda()
        if 'bb-adapt' in self.hint_adapt.type:
            for i in range(len(self.bb_adapt)):
                self.bb_adapt[i][0].weight.data.normal_().fmod_(2).mul_(0.0001).add_(0)
                self.bb_adapt[i].cuda()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        bb = self.backbone(img)
        if self.with_neck:
            x = self.neck(bb)
        return x, bb

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x, _ = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      sampling=None,
                      cls_neg_weight=-1,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x, bb = self.extract_feat(img)
        outs = self.bbox_head(x)
        cls_score, bbox_pred = outs

        kd_cfg = kwargs.get('kd_cfg')
        if 'mask-neck-one' in kd_cfg.type:
            neck_mask_batch = self.bbox_head.get_one_mask(cls_score, img_metas, gt_bboxes)
        elif 'mask-neck-gt-v1' in kd_cfg.type:
            neck_mask_batch = self.bbox_head.get_gt_mask(cls_score, img_metas, gt_bboxes, version=1)
        elif 'mask-neck-gt-v2' in kd_cfg.type:
            neck_mask_batch = self.bbox_head.get_gt_mask(cls_score, img_metas, gt_bboxes, version=2)
        elif 'mask-neck-roi-v1' in kd_cfg.type:
            # RoI based foreground, phi=0.5
            phi = kd_cfg.get('roi_phi', 0.5)
            neck_mask_batch = self.bbox_head.get_roi_mask(cls_score, img_metas, gt_bboxes, phi=phi, version=1)
        elif 'mask-neck-roi-v2' in kd_cfg.type:
            # RoI based foreground, phi=0.5, pexel value > 1
            phi = kd_cfg.get('roi_phi', 0.5)
            neck_mask_batch = self.bbox_head.get_roi_mask(cls_score, img_metas, gt_bboxes, phi=phi, version=2)
        elif 'mask-neck-roi-v3' in kd_cfg.type:
            # RoI based foreground, phi=0.5, pexel value += 1
            phi = kd_cfg.get('roi_phi', 0.5)
            neck_mask_batch = self.bbox_head.get_roi_mask(cls_score, img_metas, gt_bboxes, phi=phi, version=3)
        elif 'mask-neck-roi-v4' in kd_cfg.type:
            # RoI based background, phi=0.5
            phi = kd_cfg.get('roi_phi', 0.5)
            neck_mask_batch = self.bbox_head.get_roi_mask(cls_score, img_metas, gt_bboxes, phi=phi, version=4)
        else:
            neck_mask_batch = None

        if 'mask-bb-one' in kd_cfg.type:
            bb_mask_batch = self.bbox_head.get_bb_one_mask(cls_score, img_metas, gt_bboxes)
        elif 'mask-bb-gt-v1' in kd_cfg.type:
            bb_mask_batch = self.bbox_head.get_bb_gt_mask(cls_score, img_metas, gt_bboxes, version=1)
        elif 'mask-bb-gt-v4' in kd_cfg.type:
            bb_mask_batch = self.bbox_head.get_bb_gt_mask(cls_score, img_metas, gt_bboxes, version=4)
        else:
            bb_mask_batch = None

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses, cls_reg_targets = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        head_det = dict()
        head_det['cls_score'] = cls_score
        head_det['bbox_pred'] = bbox_pred
        head_det['neck'] = x
        head_det['backbone'] = bb
        head_det['bbox_targets'] = None
        head_det['sampling'] = None

        mask = dict()
        mask['neck_mask_batch'] = neck_mask_batch
        mask['bb_mask_batch'] = bb_mask_batch

        return losses, head_det, mask, None, cls_reg_targets

    def simple_test(self, img, img_metas, rescale=False):
        x, _ = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
