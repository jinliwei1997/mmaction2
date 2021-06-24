import copy
import os.path as osp
import torch
import numpy as np
from collections import defaultdict

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class VideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        if kwargs.get("sample_by_class", False):
            self.prob_by_class = dict()
            self.class_idx = defaultdict(list)
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        if not self.test_mode and self.sample_by_class:
            for label in self.video_infos_by_class:
                number = len(self.video_infos_by_class[label])
                self.prob_by_class[label] = np.ones(number) / number

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith(".json"):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, "r") as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                    onehot = torch.zeros(self.num_classes)
                    onehot[label] = 1.0
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(
                    dict(filename=filename, label=onehot if self.multi_class else label)
                )
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for i, item in enumerate(self.video_infos):
            label = item["label"]
            video_infos_by_class[label].append(item)
            self.class_idx[label].append(i)
        return video_infos_by_class

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            idx2 = np.random.choice(len(samples), p=self.prob_by_class[idx])
            self.prob_by_class[idx][idx2] = 0
            if np.sum(self.prob_by_class[idx]) <= 0:
                self.prob_by_class[idx] = np.ones(len(samples)) / len(samples)
            else:
                self.prob_by_class[idx] /= np.sum(self.prob_by_class[idx])
            results = copy.deepcopy(samples[idx2])
            idx = self.class_idx[idx][idx2]
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results["idx"] = idx
        results["modality"] = self.modality
        results["start_index"] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results["label"], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results["label"]] = 1.0
            results["label"] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results["modality"] = self.modality
        results["start_index"] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results["label"], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results["label"]] = 1.0
            results["label"] = onehot

        return self.pipeline(results)