import numpy as np
import argparse
import pickle

def get_gt_rank(q_feat, k_feat):
    s = np.matmul(q_feat, np.transpose(k_feat))  # [N , N]
    N = s.shape[0]
    rank = np.argsort(s)[:, ::-1]
    mask = np.repeat(np.arange(N).reshape(N, 1), axis=1, repeats=N)
    gt_rank = np.argsort(rank == mask)[:, ::-1][:, :1].reshape(N)
    return gt_rank

def cal_avg_metrics(gt_rank):
    N = gt_rank.shape[0]
    mean_rk = np.mean(gt_rank)
    median_rk = np.median(gt_rank)
    recall1 = np.sum(gt_rank < 1) / N
    recall5 = np.sum(gt_rank < 5) / N
    recall10 = np.sum(gt_rank < 10) / N
    return mean_rk, median_rk, recall1, recall5, recall10

def eval_retrieval_metrics(v_feat, t_feat, split_type = 'full', metrics = ['mean_rk', 'recall1', 'recall5', 'recall10']):
    tot = len(v_feat)
    if split_type == 'full':
        split = [[i for i in range(len(v_feat))]]
    else:
        raise NotImplementedError

    v_t_gt_rank=[]
    t_v_gt_rank=[]
    for subset in split:
        sub_v_feat = [v_feat[i] for i in subset]
        sub_t_feat = [t_feat[i] for i in subset]
        v_t_gt_rank.extend(get_gt_rank(np.array(sub_v_feat), np.array(sub_t_feat)))
        t_v_gt_rank.extend(get_gt_rank(np.array(sub_t_feat), np.array(sub_v_feat)))

    eval_results = {}


    mean_rk, median_rk, recall1, recall5, recall10 = cal_avg_metrics(np.array(v_t_gt_rank))
    if 'mean_rk' in metrics:
        eval_results['v_t_mean_rk'] = mean_rk
    if 'median_rk' in metrics:
        eval_results['v_t_mean_rk'] = median_rk
    if 'recall1' in metrics:
        eval_results['v_t_recall1'] = recall1
    if 'recall5' in metrics:
        eval_results['v_t_recall5'] = recall5
    if 'recall10' in metrics:
        eval_results['v_t_recall10'] = recall10

    # print('---V to T retrieval---\n')
    # print(f'vt_mean_rk\t{mean_rk:.4f}\nvt_median_rk\t{median_rk:.4f}\nvt_recall1\t{recall1:.4f}\nvt_recall5\t{recall5:.4f}\nvt_recall10\t{recall10:.4f}\n')

    mean_rk, median_rk, recall1, recall5, recall10 = cal_avg_metrics(np.array(t_v_gt_rank))
    if 'mean_rk' in metrics:
        eval_results['t_v_mean_rk'] = mean_rk
    if 'median_rk' in metrics:
        eval_results['t_v_mean_rk'] = median_rk
    if 'recall1' in metrics:
        eval_results['t_v_recall1'] = recall1
    if 'recall5' in metrics:
        eval_results['t_v_recall5'] = recall5
    if 'recall10' in metrics:
        eval_results['t_v_recall10'] = recall10

    # print('---T to V retrieval---\n')
    # print(f'vt_mean_rk\t{mean_rk:.4f}\nvt_median_rk\t{median_rk:.4f}\nvt_recall1\t{recall1:.4f}\nvt_recall5\t{recall5:.4f}\nvt_recall10\t{recall10:.4f}\n\n')

    return eval_results