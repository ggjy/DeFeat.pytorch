from ..builder import DETECTORS
from .single_stage_kd import SingleStageDetectorKD


@DETECTORS.register_module()
class RetinaNetKD(SingleStageDetectorKD):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 hint_adapt=dict()):
        super(RetinaNetKD, self).__init__(
            backbone,
            neck,
            bbox_head,
            train_cfg,
            test_cfg,
            pretrained,
            hint_adapt=hint_adapt)
