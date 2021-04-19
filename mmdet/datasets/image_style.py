import os.path as osp

import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset
from PIL import Image


@DATASETS.register_module
class ImageDataset(CustomDataset):

    def __init__(self, min_size=None, train_thr=0.5, **kwargs):
        super(ImageDataset, self).__init__(**kwargs)
        
        CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
           'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
           'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
           'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

        # for COCO datasst
        self.cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                        80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.min_size = min_size
        self.train_thr = train_thr

    def load_annotations(self, ann_file):
        data_infos = []
        self.img_ids = mmcv.list_from_file(ann_file)
        import json
        import os
        if 'selected_data' in ann_file:
            bbox_file = ann_file.replace('selected_data.txt', 'selected_results.bbox.json')
        elif 'selected_v2' in ann_file:
            bbox_file = ann_file.replace('selected_v2.txt', 'selected_v2_results.bbox.json')
        elif 'selected_v3' in ann_file:
            bbox_file = ann_file.replace('selected_v3.txt', 'selected_v3_results.bbox.json')
        elif 'imagenet_train.txt' in ann_file:
            bbox_file = ann_file.replace('imagenet_train.txt', 'imagenet_train.bbox.json')
        elif 'unlabeled2017' in ann_file:
            self.img_ids = mmcv.load(ann_file)['images']
            bbox_file = ann_file.replace('unlabeled2017.json', 'unlabeled2017.bbox.json')
        elif 'imagenet_selected_118k' in ann_file:
            bbox_file = ann_file.replace('imagenet_selected_118k.txt', 'imagenet_selected_118k.bbox.json')
        else:
            raise NotImplementedError('No annotations info.')
        
        if os.path.isfile(bbox_file):
            json_file = open(bbox_file).read()
            self.ann_info = json.loads(json_file)
        else:
            self.ann_info = []

        if 'unlabeled2017' in ann_file:
            # coco unlabeled2017 json
            for i, info in enumerate(self.img_ids):
                filename = info['file_name']
                height = info['height']
                width = info['width']
                img_id = filename.split(".")[0]
                self.img_ids[i] = filename
                data_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))
            self.info_dict = dict()
            for i in range(len(self.ann_info)):
                filename = self.ann_info[i]['image_id']
                if filename in self.info_dict.keys():
                    self.info_dict[filename].append(i)
                else:
                    self.info_dict[filename] = [i]
        else:
            # imagenet style image
            for i, filename in enumerate(self.img_ids):
                if filename.count(" ") == 1:
                    imagenet_category = filename.split(" ")[-1] # string
                    filename = filename.split(" ")[0]
                    self.img_ids[i] = filename
                if filename.count("/") >= 1:
                    img_id = filename.split("/")[-1].split(".")[0]
                    # parent_dir = filename[0:-len(filename.split("/")[-1]) - 1]
                    # filename = filename.split("/")[-1]
                else:
                    # parent_dir = ""
                    img_id = filename.split(".")[0]
                img = Image.open(osp.join(self.img_prefix, filename))
                width = int(img.size[0])
                height = int(img.size[1])
                data_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))
            self.info_dict = dict()
            for i in range(len(self.ann_info)):
                filename = self.ann_info[i]['image_id']
                if filename in self.info_dict.keys():
                    self.info_dict[filename].append(i)
                else:
                    self.info_dict[filename] = [i]

        return data_infos

    def get_ann_info(self, idx):
        filename = self.data_infos[idx]['filename']
        ann_info = []
        if filename in self.info_dict.keys():
            for i in self.info_dict[filename]:
                ann_info.append(self.ann_info[i])
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):     
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(value['image_id'] for value in self.ann_info)
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
        """
        gt_bboxes = []
        gt_labels = []

        if ann_info:
            for i, ann in enumerate(ann_info):
                score = ann['score']
                x1, y1, w, h = ann['bbox']

                if score > self.train_thr:
                    bbox = [x1, y1, x1 + w, y1 + h]

                    gt_bboxes.append(bbox)
                    gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=None,
            seg_map=None)

        return ann

    def xyxy2xywh(self, bbox):
        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results, thr=0.):
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    if float(bboxes[i][4]) > thr:
                        data = dict()
                        data['image_id'] = img_id
                        data['bbox'] = self.xyxy2xywh(bboxes[i])
                        data['score'] = float(bboxes[i][4])
                        data['category_id'] = self.cat_ids[label]
                        json_results.append(data)
        return json_results

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix, thr=0.):
        """Dump the detection results to a json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results, thr=thr)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = '{}.{}.json'.format(outfile_prefix, 'bbox')
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'bbox')
            result_files['segm'] = '{}.{}.json'.format(outfile_prefix, 'segm')
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = '{}.{}.json'.format(
                outfile_prefix, 'proposal')
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[i])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, thr=0., **kwargs):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix, thr=thr)
        return result_files, tmp_dir
