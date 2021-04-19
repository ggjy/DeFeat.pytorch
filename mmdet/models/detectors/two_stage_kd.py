import torch
import torch.nn as nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from .test_mixins import RPNTestMixin


@DETECTORS.register_module()
class TwoStageDetectorKD(BaseDetector, RPNTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 hint_adapt=dict(type='neck'),
                 pretrained=None):
        super(TwoStageDetectorKD, self).__init__()
        self.backbone = build_backbone(backbone)
        self.hint_adapt = hint_adapt

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

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
                        nn.Sequential(),
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
                        nn.ReLU(),
                    )
                )
            self.bb_adapt = nn.ModuleList(self.bb_adapt)

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetectorKD, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)
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
        outs = ()
        # backbone
        x, _ = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      sampling=None,
                      cls_neg_weight=-1,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, bb = self.extract_feat(img)
        neck_mask_batch = None
        bb_mask_batch = None

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)  # cls_score, bbox_pred
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas)

            kd_cfg = kwargs.get('kd_cfg')
            if 'mask-neck-one' in kd_cfg.type:
                # All regions in DeFeat
                neck_mask_batch = self.rpn_head.get_one_mask(rpn_outs[0], img_metas, gt_bboxes)
            elif 'mask-neck-gt' in kd_cfg.type:
                neck_mask_batch = self.rpn_head.get_gt_mask(rpn_outs[0], img_metas, gt_bboxes)
            elif 'mask-neck-roi' in kd_cfg.type:
                # FGFI
                # RoI \phi=0.5
                phi = kd_cfg.get('roi_phi', 0.5)
                neck_mask_batch = self.rpn_head.get_roi_mask(rpn_outs[0], img_metas, gt_bboxes, phi=phi)

            if 'mask-bb-one' in kd_cfg.type:
                bb_mask_batch = self.rpn_head.get_bb_one_mask(rpn_outs[0], img_metas, gt_bboxes)
            elif 'mask-bb-gt' in kd_cfg.type:
                bb_mask_batch = self.rpn_head.get_bb_gt_mask(rpn_outs[0], img_metas, gt_bboxes)

            rpn_losses, rpn_targets = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, cls_neg_weight=cls_neg_weight)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(
                *rpn_outs, img_metas, cfg=proposal_cfg)
            
        else:
            proposal_list = proposals
            
        roi_losses, cls_score, bbox_pred, bbox_targets, \
        sampling_results = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, sampling=sampling, cls_neg_weight=cls_neg_weight)
        losses.update(roi_losses)

        head_det = dict()
        head_det['cls_score'] = cls_score
        head_det['bbox_pred'] = bbox_pred
        head_det['bbox_targets'] = bbox_targets
        head_det['sampling'] = sampling_results
        head_det['neck'] = x
        head_det['backbone'] = bb
        head_det['img_metas'] = img_metas
        head_det['gt_bboxes'] = gt_bboxes
        head_det['gt_labels'] = gt_labels

        mask = dict()
        mask['neck_mask_batch'] = neck_mask_batch
        mask['bb_mask_batch'] = bb_mask_batch

        return losses, head_det, mask, rpn_outs, rpn_targets

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x, _ = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.async_test_rpn(x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x, _ = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x, _ = self.extract_feats(imgs)
        proposal_list = self.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
