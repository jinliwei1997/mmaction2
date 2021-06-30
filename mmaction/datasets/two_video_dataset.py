import os.path as osp
import copy

import torch
import numpy as np

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class TwoVideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file contains two anno files, each with multiple lines, and each line indicates
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

    def __init__(
            self, ann_file, pipeline, start_index=0, noise_ratio=2, gpus=1, **kwargs
    ):
        # make sure the gpus is correct

        self.noise_ratio = noise_ratio + 1
        self.video_infos1 = []
        self.video_infos2 = []
        self.video_num1, self.video_num2 = 0, 0
        self.gpus = gpus
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        if not self.test_mode:
            self.video_idxes1_per_gpu = []
            self.video_idxes2_per_gpu = []
            self.indices1 = list(range(self.video_num1))
            self.indices2 = list(range(self.video_num2))
            for i in range(gpus):
                self.video_idxes1_per_gpu.append(
                    self.indices1[i : self.video_num1 : gpus]
                )
                self.video_idxes2_per_gpu.append(
                    self.indices2[i : self.video_num2 : gpus]
                )
            self.count_per_gpu = [0] * gpus

    def load_annotations(self):
        """Load annotation file to get video information."""
        if not self.test_mode:
            anno1, anno2 = self.ann_file.split()

            with open(anno1, "r") as fin:
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
                    self.video_infos1.append(
                        dict(
                            filename=filename,
                            label=onehot if self.multi_class else label,
                        )
                    )
            self.video_num1 = len(self.video_infos1)

            with open(anno2, "r") as fin:
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
                    self.video_infos2.append(
                        dict(
                            filename=filename,
                            label=onehot if self.multi_class else label,
                        )
                    )
            self.video_num2 = len(self.video_infos2)

            return self.video_infos1 + self.video_infos2
        else:
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
                        dict(
                            filename=filename,
                            label=onehot if self.multi_class else label,
                        )
                    )
            return video_infos

    def prepare_train_frames(self, idx):
        gpu_idx = idx % self.gpus
        if self.count_per_gpu[gpu_idx] == 0:  # sample video_infos1
            if len(self.video_idxes1_per_gpu[gpu_idx]) == 0:
                self.video_idxes1_per_gpu[gpu_idx] = self.indices1[
                                                     gpu_idx : self.video_num1 : self.gpus
                                                     ]
            video_idx1 = np.random.choice(
                self.video_idxes1_per_gpu[gpu_idx], 1, replace=False
            )[0]
            self.video_idxes1_per_gpu[gpu_idx].remove(video_idx1)
            results = copy.deepcopy(self.video_infos1[video_idx1])
        else:  # sample video_infos2
            if len(self.video_idxes2_per_gpu[gpu_idx]) == 0:
                self.video_idxes2_per_gpu[gpu_idx] = self.indices2[
                                                     gpu_idx : self.video_num2, self.gpus
                                                     ]
            video_idx2 = np.random.choice(
                self.video_idxes2_per_gpu[gpu_idx], 1, replace=False
            )[0]
            self.video_idxes2_per_gpu[gpu_idx].remove(video_idx2)
            results = copy.deepcopy(self.video_infos2[video_idx2])
        self.count_per_gpu[gpu_idx] = (
                                              self.count_per_gpu[gpu_idx] + 1
                                      ) % self.noise_ratio

        results["modality"] = self.modality
        results["start_index"] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results["label"], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results["label"]] = 1.0
            results["label"] = onehot

        return self.pipeline(results)