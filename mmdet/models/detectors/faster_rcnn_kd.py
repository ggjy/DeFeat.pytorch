from ..builder import DETECTORS
from .two_stage_kd import TwoStageDetectorKD


@DETECTORS.register_module()
class FasterRCNNKD(TwoStageDetectorKD):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 hint_adapt='neck',
                 pretrained=None):
        super(FasterRCNNKD, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            hint_adapt=hint_adapt,
            pretrained=pretrained)
