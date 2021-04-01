import copy
import os.path as osp

import numpy as np
import torch

from .base import BaseDataset
from .registry import DATASETS
import warnings
from mmcv.utils import print_log

@DATASETS.register_module()
class Mp4TextDataset(BaseDataset):
    """VideoText dataset for matcher.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the path of the corresponding text annotation, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 some/text1
        some/directory-2 122 some/text2
        some/directory-3 258 some/text3
        some/directory-4 234 some/text4
        some/directory-5 295 some/text5
        some/directory-6 121 some/text6


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float | None): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: None.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=None):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power)

    def load_annotations(self):
        """Load annotation file to get video and text information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line = line.strip()
                video_info = {}

                video_info['filename'] = line.split(' ##$$## ')[0].rstrip()
                video_info['text_path'] = line.split(' ##$$## ')[1].rstrip()

                video_infos.append(video_info)

        return video_infos

    def evaluate(self,
                 results,
                 metrics=['vt_retrieval_metrics_full', 'tv_retrieval_metrics_full'],
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['vt_retrieval_metrics_full', 'tv_retrieval_metrics_full']

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        # for metric in metrics:
        #     msg = f'Evaluating {metric} ...'
        #     if logger is None:
        #         msg = '\n' + msg
        #     print_log(msg, logger=logger)
        # vt_recall5_full\t{recall5:.4f}\nvt_recall10_full\t{recall10:.4f}'
        #     if metric == 'vt_retrieval_metrics_full':
        #         v_feat = np.array([result[0] for result in results])
        #         t_feat = np.array([result[1] for result in results])
        #         assert t_feat.shape[1] == 1  # 1 video vs. 1 text
        #         mean_rk, median_rk, recall1, recall5, recall10 = eval_retrieval_metrics(v_feat, t_feat.reshape(t_feat[0], -1))
        #
        #         eval_results['vt_mean_rk_full'] = mean_rk
        #         eval_results['vt_median_rk_full'] = median_rk
        #         eval_results['vt_recall1_full'] = recall1
        #         eval_results['vt_recall5_full'] = recall5
        #         eval_results['vt_recall10_full'] = recall10
        #
        #         log_msg = f'\nvt_mean_rk_full\t{mean_rk:.4f}\nvt_median_rk_full\t{median_rk:.4f}\nvt_recall1_full\t{recall1:.4f}\n
        #         print_log(log_msg, logger=logger)
        #
        #     if metric == 'tv_retrieval_metrics_full':
        #         v_feat = np.array([result[0] for result in results])
        #         t_feat = np.array([result[1] for result in results])
        #         assert t_feat.shape[1] == 1 # 1 video vs. 1 text
        #         mean_rk, median_rk, recall1, recall5, recall10 = eval_retrieval_metrics(t_feat.reshape(t_feat[0],-1),v_feat)
        #
        #         eval_results['tv_mean_rk_full'] = mean_rk
        #         eval_results['tv_median_rk_full'] = median_rk
        #         eval_results['tv_recall1_full'] = recall1
        #         eval_results['tv_recall5_full'] = recall5
        #         eval_results['tv_recall10_full'] = recall10
        #
        #         log_msg = f'\ntv_mean_rk_full\t{mean_rk:.4f}\ntv_median_rk_full\t{median_rk:.4f}\ntv_recall1_full\t{recall1:.4f}\ntv_recall5_full\t{recall5:.4f}\ntv_recall10_full\t{recall10:.4f}'
        #         print_log(log_msg, logger=logger)

        return eval_results

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index


        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        if self.sample_by_class:
            # Then, the idx is the class index
            samples = self.video_infos_by_class[idx]
            results = copy.deepcopy(np.random.choice(samples))
        else:
            results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

def eval_retrieval_metrics(q_feat, k_feat):
    s = np.matmul(q_feat, np.transpose(k_feat)) # [N , N]
    N = s.shape[0]
    rank = np.argsort(s)[:,::-1]
    mask = np.repeat(np.arange(N).reshape(N, 1), axis=1, repeats=N)
    gt_rank = np.argsort(rank == mask)[:, ::-1][:, :1].reshape(N)
    mean_rk = np.mean(gt_rank)
    median_rk = np.median(gt_rank)
    recall1 = np.sum(gt_rank < 1) / N
    recall5 = np.sum(gt_rank < 5) / N
    recall10 = np.sum(gt_rank < 10) / N
    return mean_rk, median_rk, recall1, recall5, recall10