## Modified by Jianyuan Guo, guojian.guo@huawei.com
## 2020/05
#import matplotlib
#matplotlib.use('Agg')
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner, Runner_kd

from mmdet.core import (DistEvalHook, DistOptimizerHook, EvalHook,
                        Fp16OptimizerHook, build_optimizer)
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def KLDivergenceLoss(y, teacher_scores, mask=None, T=1):
    if mask is not None:
        if mask.sum() > 0:
            p = F.log_softmax(y/T, dim=1)[mask]
            q = F.softmax(teacher_scores/T, dim=1)[mask]
            l_kl = F.kl_div(p, q, reduce=False)
            loss = torch.sum(l_kl)
            loss = loss / mask.sum()
        else:
            loss = torch.Tensor([0]).cuda()
    else:
        p = F.log_softmax(y/T, dim=1)
        q = F.softmax(teacher_scores/T, dim=1)
        l_kl = F.kl_div(p, q, reduce=False)
        loss = l_kl.sum() / l_kl.size(0)
    return loss * T**2


def BCELoss(y, teacher_scores, mask):
    p = F.softmax(y, dim=1)[mask]
    q = F.softmax(teacher_scores, dim=1)[mask]
    loss = F.binary_cross_entropy(p, q.detach()) * 10.0
    return loss


def l1loss(pred_s, pred_t, target):
    assert pred_s.size() == pred_t.size() == target.size() and target.numel() > 0
    loss_s_t = torch.abs(pred_s - pred_t).sum(1) / 4.0
    loss_s_gt = torch.abs(pred_s - target).sum(1) / 4.0
    loss = loss_s_t[loss_s_t<=loss_s_gt].sum() + loss_s_gt[loss_s_gt<loss_s_t].sum()
    return loss / target.size(0)


def l1rpnloss(pred_s, pred_t, target, weights):
    assert pred_s.size() == pred_t.size() == target.size()
    loss_s_t = torch.abs(pred_s * weights - pred_t * weights).sum(1) / 4.0
    loss_s_gt = torch.abs(pred_s * weights - target * weights).sum(1) / 4.0
    loss = loss_s_t[loss_s_t<=loss_s_gt].sum() + loss_s_gt[loss_s_gt<loss_s_t].sum()
    return loss, weights.sum()/4


def mseloss(pred_s, pred_t, target, weights):
    if weights is not None:
        pred_t = pred_t[weights.type(torch.bool)]
        pred_s = pred_s[weights.type(torch.bool)]
        if weights.sum() > 0:
            pred_s = pred_s.sigmoid()
            pred_t = pred_t.sigmoid()
            loss = F.mse_loss(pred_s, pred_t, reduction='none')
            return loss.sum(), weights.sum()
        else:
            return 0., 0.
    else:
        pred_s = pred_s.sigmoid()
        pred_t = pred_t.sigmoid()
        loss = F.mse_loss(pred_s, pred_t, reduction='none')
        return loss.sum(), loss.size(0)


def batch_processor_kd(model, model_t, data, train_mode, kd_warm=dict(), kd_decay=1., epoch=0, **kwargs):
    
    kd_cfg = kwargs.get('kd_cfg')
    if 'clsNegIgnore' in kd_cfg.type:
        cls_neg_weight = kd_cfg.get('cls_neg_weight', -1) / (kd_cfg.get('cls_neg_weight_decay', 5)**epoch)
    else:
        cls_neg_weight = -1
    if 'SampleSTU' in kd_cfg.type:
        losses, head_det_s, mask_s, rpn_outs_s, rpn_targets = model(
            **data, cls_neg_weight=cls_neg_weight, **kwargs)
        neck_mask_batch = mask_s['neck_mask_batch']
        bb_mask_batch = mask_s['bb_mask_batch']
        neck_feat = head_det_s['neck']
        bb_feat = head_det_s['backbone']
        cls_score_s = head_det_s['cls_score']
        bbox_pred_s = head_det_s['bbox_pred'] 
        bbox_target_gt = head_det_s['bbox_targets']
        sampling_results = head_det_s['sampling']

        if model_t is not None:
            _, head_det_t, _, rpn_outs_t, _ = model_t(
                **data, sampling=sampling_results, cls_neg_weight=cls_neg_weight, **kwargs)
            neck_feat_t = head_det_t['neck']
            bb_feat_t = head_det_t['backbone']
            cls_score_t = head_det_t['cls_score']
            bbox_pred_t = head_det_t['bbox_pred']
    elif 'SampleDUO' in kd_cfg.type:
        losses, head_det_s, mask_s, rpn_outs_s, rpn_targets = model(
            **data, cls_neg_weight=cls_neg_weight, **kwargs)
        neck_mask_batch = mask_s['neck_mask_batch']
        bb_mask_batch = mask_s['bb_mask_batch']
        neck_feat = head_det_s['neck']
        bb_feat = head_det_s['backbone']
        cls_score_ss = head_det_s['cls_score']
        bbox_pred_ss = head_det_s['bbox_pred'] 
        bbox_target_gt = head_det_s['bbox_targets']
        sampling_results = head_det_s['sampling']

        _, head_det_t, _, rpn_outs_t, rpn_targets_t = model_t(
            **data, cls_neg_weight=cls_neg_weight, **kwargs)
        neck_feat_t = head_det_t['neck']
        bb_feat_t = head_det_t['backbone']
        cls_score_tt = head_det_t['cls_score']
        bbox_pred_tt = head_det_t['bbox_pred']
        bbox_target_gt_te = head_det_t['bbox_targets']
        sampling_results_t = head_det_t['sampling']
        
        _, cls_score_st, bbox_pred_st, _, _ = model.module.roi_head.forward_train(
            head_det_s['neck'], head_det_s['img_metas'], None, 
            head_det_s['gt_bboxes'], head_det_s['gt_labels'], None, None, 
            sampling=sampling_results_t, cls_neg_weight=cls_neg_weight)

        _, cls_score_ts, bbox_pred_ts, _, _ = model_t.module.roi_head.forward_train(
            head_det_t['neck'], head_det_t['img_metas'], None, 
            head_det_t['gt_bboxes'], head_det_t['gt_labels'], None, None, 
            sampling=sampling_results, cls_neg_weight=cls_neg_weight)
    else:
        # RoI samples from teacher network (SampleTEA)
        losses, head_det_s, mask_s, rpn_outs_s, _ = model(
            **data, cls_neg_weight=cls_neg_weight, **kwargs)
        neck_mask_batch = mask_s['neck_mask_batch']
        bb_mask_batch = mask_s['bb_mask_batch']
        neck_feat = head_det_s['neck']
        bb_feat = head_det_s['backbone']
        bbox_target_gt = head_det_s['bbox_targets'] 

        if model_t is not None:
            # model_t return: losses, head_det, mask, rpn_outs, rpn_targets
            _, head_det_t, _, rpn_outs_t, rpn_targets = model_t(
                **data, cls_neg_weight=cls_neg_weight, **kwargs)
            neck_feat_t = head_det_t['neck']
            bb_feat_t = head_det_t['backbone']
            sampling_results = head_det_t['sampling']
            cls_score_t = head_det_t['cls_score']

            if hasattr(model.module, 'roi_head'):
                _, cls_score_s, bbox_pred_s, _, _ = model.module.roi_head.forward_train(
                    head_det_s['neck'], head_det_s['img_metas'], None, 
                    head_det_s['gt_bboxes'], head_det_s['gt_labels'], None, None, 
                    sampling=sampling_results, cls_neg_weight=cls_neg_weight) 

    if 'head-cls' in kd_cfg.type:
        labels = bbox_target_gt[0]  # size 2048

        if 'BCE-fore' in kd_cfg.type:
            # only distill foreground, BCE
            pos_inds = (labels >= 0) & (labels < labels.max())
            if pos_inds.sum()==0:
                loss_hcls = torch.Tensor([0]).cuda()
            else:
                loss_hcls = BCELoss(cls_score_s, cls_score_t, pos_inds)
        elif 'BCE-correct' in kd_cfg.type:
            # only distill correct teacher (include background), BCE
            indice_t = torch.max(cls_score_t, 1)[1]  # size 2048
            pos_inds = labels == indice_t
            if pos_inds.sum()==0:
                loss_hcls = torch.Tensor([0]).cuda()
            else:
                loss_hcls = BCELoss(cls_score_s, cls_score_t, pos_inds)
        elif 'KL-fore' in kd_cfg.type:
            # only distill foreground, KL
            pos_inds = (labels >= 0) & (labels < labels.max())
            loss_hcls = KLDivergenceLoss(cls_score_s, cls_score_t, pos_inds, T=kd_cfg.head_cls_T)
        elif 'KL-cfore' in kd_cfg.type:
            pos_inds = (labels < labels.max()) & (labels == torch.max(cls_score_t, 1)[1])
            loss_hcls = KLDivergenceLoss(cls_score_s, cls_score_t, pos_inds, T=kd_cfg.head_cls_T)
        elif 'KL-back' in kd_cfg.type:
            # only distill background KL
            pos_inds = labels == labels.max()
            if 'SampleDUO' in kd_cfg.type:
                loss_hcls = KLDivergenceLoss(cls_score_ss, cls_score_ts, pos_inds, T=kd_cfg.head_cls_T)
                pos_inds = bbox_target_gt_te[0] == bbox_target_gt_te[0].max()
                loss_hcls_sampleTe = KLDivergenceLoss(cls_score_st, cls_score_tt, pos_inds, T=kd_cfg.head_cls_T)
            else:
                loss_hcls = KLDivergenceLoss(cls_score_s, cls_score_t, pos_inds, T=kd_cfg.head_cls_T)
        elif 'KL-decouple' in kd_cfg.type:
            pos_inds = labels < labels.max()
            loss_hcls = KLDivergenceLoss(cls_score_s, cls_score_t, pos_inds, T=kd_cfg.head_cls_T)
            pos_inds = labels == labels.max()
            loss_hcls_back = KLDivergenceLoss(cls_score_s, cls_score_t, pos_inds, T=kd_cfg.head_cls_T)
        elif 'KL-all' in kd_cfg.type:
            # distill all proposals, KL
            pos_inds = labels >= 0
            if 'SampleDUO' in kd_cfg.type:
                loss_hcls = KLDivergenceLoss(cls_score_ss, cls_score_ts, pos_inds, T=kd_cfg.head_cls_T)
                pos_inds = bbox_target_gt_te[0] >= 0
                loss_hcls_sampleTe = KLDivergenceLoss(cls_score_st, cls_score_tt, pos_inds, T=kd_cfg.head_cls_T)
            else:
                loss_hcls = KLDivergenceLoss(cls_score_s, cls_score_t, pos_inds, T=kd_cfg.head_cls_T)
        else:
            # only distill correct teacher (include background), KL
            indice_t = torch.max(cls_score_t, 1)[1]  # size 2048
            pos_inds = labels == indice_t
            loss_hcls = KLDivergenceLoss(cls_score_s, cls_score_t, pos_inds, T=kd_cfg.head_cls_T)

        loss_hcls *= kd_cfg.head_cls_w
        if 'decay' in kd_cfg.type:
            loss_hcls *= kd_decay
        if kd_warm.get('head-cls', False):
            loss_hcls *= 0
        losses['losskd_hcls'] = loss_hcls
        if 'SampleDUO' in kd_cfg.type:
            loss_hcls_sampleTe *= kd_cfg.head_cls_w
            if 'decay' in kd_cfg.type:
                loss_hcls_sampleTe *= kd_decay
            losses['losskd_hcls_sampleTe'] = loss_hcls_sampleTe
        if 'KL-decouple' in kd_cfg.type:
            loss_hcls_back *= kd_cfg.head_cls_back_w
            if 'decay' in kd_cfg.type:
                loss_hcls_back *= kd_decay
            losses['losskd_hcls_back'] = loss_hcls_back

    # kd: neck imitation w/ or w/o adaption layer
    if 'neck' in kd_cfg.type:
        losskd_neck = torch.Tensor([0]).cuda()
        if 'distribution-kl' in kd_cfg.type:
            loss_dist_neck = torch.Tensor([0]).cuda()
        if 'neck-decouple' in kd_cfg.type:
            losskd_neck_back = torch.Tensor([0]).cuda()
        for i, _neck_feat in enumerate(neck_feat):
            mask_hint = neck_mask_batch[i]
            mask_hint = mask_hint.unsqueeze(1).repeat(1, _neck_feat.size(1), 1, 1)
            norms = max(1.0, mask_hint.sum() * 2)
            if 'neck-adapt' in kd_cfg.type and hasattr(model.module, 'neck_adapt'):
                neck_feat_adapt = model.module.neck_adapt[i](_neck_feat)
            else:
                neck_feat_adapt = _neck_feat
            
            if 'pixel-wise' in kd_cfg.type:               
                if 'L1' in kd_cfg.type:
                    diff = torch.abs(neck_feat_adapt - neck_feat_t[i])
                    loss = torch.where(diff < 1.0, diff, diff**2)
                    losskd_neck += (loss * mask_hint).sum() / norms
                elif 'Div' in kd_cfg.type:
                    losskd_neck += (torch.pow(1 - neck_feat_adapt / (neck_feat_t[i] + 1e-8), 2) * mask_hint).sum() / norms
                elif 'neck-decouple' in kd_cfg.type:
                    norms_back = max(1.0, (1 - mask_hint).sum() * 2)
                    losskd_neck_back += (torch.pow(neck_feat_adapt - neck_feat_t[i], 2) * 
                                        (1 - mask_hint)).sum() / norms_back
                    losskd_neck += (torch.pow(neck_feat_adapt - neck_feat_t[i], 2) * mask_hint).sum() / norms
                else:
                    losskd_neck = losskd_neck + (torch.pow(neck_feat_adapt - neck_feat_t[i], 2) * 
                                                 mask_hint).sum() / norms

        if 'pixel-wise' in kd_cfg.type:
            losskd_neck = losskd_neck / len(neck_feat)
            losskd_neck = losskd_neck * kd_cfg.hint_neck_w
            if 'decay' in kd_cfg.type:
                losskd_neck *= kd_decay
            if kd_warm.get('hint', False):
                losskd_neck *= 0.
            losses['losskd_neck'] = losskd_neck

        if 'neck-decouple' in kd_cfg.type:
            losskd_neck_back = losskd_neck_back / len(neck_feat)
            losskd_neck_back = losskd_neck_back * kd_cfg.hint_neck_back_w
            if 'decay' in kd_cfg.type:
                losskd_neck_back *= kd_decay
            if kd_warm.get('hint', False):
                losskd_neck_back *= 0.
            losses['losskd_neck_back'] = losskd_neck_back

    # kd: backbone imitation w/ or w/o adaption layer
    if 'bb' in kd_cfg.type and hasattr(kd_cfg, 'bb_indices'):
        losskd_bb = torch.Tensor([0]).cuda()
        if 'bb-decouple' in kd_cfg.type:
            losskd_bb_back = torch.Tensor([0]).cuda()
        mask_hint = bb_mask_batch.unsqueeze(1)
        for i, indice in enumerate(kd_cfg.bb_indices):
            if 'bb-adapt' in kd_cfg.type and hasattr(model.module, 'bb_adapt'):
                bb_feat_adapt = model.module.bb_adapt[i](bb_feat[indice])
            else:
                bb_feat_adapt = bb_feat[indice]
            c, h, w = bb_feat_adapt.shape[1:]
            mask_bb = F.interpolate(mask_hint, size=[h, w], mode="nearest").repeat(1, c, 1, 1)
            norms = max(1.0, mask_bb.sum() * 2)
            if 'bb-decouple' in kd_cfg.type:
                losskd_bb += (torch.pow(bb_feat_adapt - bb_feat_t[indice], 2) * mask_bb).sum() / norms
                norms_back = max(1, (1 - mask_bb).sum() * 2)
                losskd_bb_back += (torch.pow(bb_feat_adapt - bb_feat_t[indice], 2) * (1 - mask_bb)).sum() / norms_back
            else:
                losskd_bb += (torch.pow(bb_feat_adapt - bb_feat_t[indice], 2) * mask_bb).sum() / norms

        losskd_bb /= len(kd_cfg.bb_indices)
        losskd_bb *= kd_cfg.hint_bb_w
        if 'bb-decouple' in kd_cfg.type:
            losskd_bb_back /= len(kd_cfg.bb_indices)
            losskd_bb_back *= kd_cfg.hint_bb_back_w
        if 'decay' in kd_cfg.type:
            losskd_bb *= kd_decay
            if 'bb-decouple' in kd_cfg.type:
                losskd_bb_back *= kd_decay
        if kd_warm.get('hint', False):
            losskd_bb *= 0
            if 'bb-decouple' in kd_cfg.type:
                losskd_bb_back *= 0
        losses['losskd_bb'] = losskd_bb
        if 'bb-decouple' in kd_cfg.type:
            losses['losskd_bb_back'] = losskd_bb_back

    loss, log_vars = parse_losses(losses)
    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model,
        batch_processor,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def train_detector_kd(model,
                      model_t,
                      dataset,
                      cfg,
                      distributed=False,
                      validate=False,
                      timestamp=None,
                      meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if model_t is not None:
            model_t = MMDataParallel(
                model_t.cuda(),
                device_ids=[torch.cuda.current_device()])
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if model_t is not None:
            model_t = MMDataParallel(
                model_t.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner_kd(
        model,
        model_t,
        batch_processor_kd,
        optimizer,
        cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, kd_cfg=cfg.model.hint_adapt)