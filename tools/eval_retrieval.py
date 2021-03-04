import numpy as np
import argparse
import pickle
def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('--v_t_feature', help='V_T feature pickle file path')
    parser.add_argument('--random_1000_split_list', default=None, help='random 1000 split list pickle file path')
    parser.add_argument('--inter_class_split_list', default=None, help='inter class split list pickle file path')
    parser.add_argument('--intra_class_split_list', default=None, help='intra class split list pickle file path')
    args = parser.parse_args()
    return args

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

def eval_retrieval_metrics(v_feat, t_feat, split_name, split):
    print(f'Eval {split_name}:')
    print(f'Total videos of val set: {v_feat.shape[0]}')
    print(f'Subsets: {len(split)} avg videos per subset: {v_feat.shape[0]/len(split)}')

    v_t_gt_rank=[]
    t_v_gt_rank=[]
    for subset in split:
        sub_v_feat = [v_feat[i] for i in subset]
        sub_t_feat = [t_feat[i] for i in subset]
        v_t_gt_rank.extend(get_gt_rank(np.array(sub_v_feat), np.array(sub_t_feat)))
        t_v_gt_rank.extend(get_gt_rank(np.array(sub_t_feat), np.array(sub_v_feat)))

    mean_rk, median_rk, recall1, recall5, recall10 = cal_avg_metrics(np.array(v_t_gt_rank))
    print('---V to T retrieval---\n')
    print(f'vt_mean_rk\t{mean_rk:.4f}\nvt_median_rk\t{median_rk:.4f}\nvt_recall1\t{recall1:.4f}\nvt_recall5\t{recall5:.4f}\nvt_recall10\t{recall10:.4f}\n')

    mean_rk, median_rk, recall1, recall5, recall10 = cal_avg_metrics(np.array(t_v_gt_rank))
    print('---T to V retrieval---\n')
    print(f'vt_mean_rk\t{mean_rk:.4f}\nvt_median_rk\t{median_rk:.4f}\nvt_recall1\t{recall1:.4f}\nvt_recall5\t{recall5:.4f}\nvt_recall10\t{recall10:.4f}\n\n')

def main():
    args = parse_args()
    results = pickle.load(open(args.v_t_feature, 'rb'))
    v_feat = np.array([result[0] for result in results])
    t_feat = np.array([result[1] for result in results])
    t_feat = t_feat.reshape(t_feat.shape[0], -1)
    if args.random_1000_split_list is None:
        random_1000_split_list = [range(i, min(i+1000, v_feat.shape[0])) for i in range(0, v_feat.shape[0], 1000)]
    eval_retrieval_metrics(v_feat, t_feat, 'random_1000_split', random_1000_split_list)
    if args.inter_class_split_list is not None:
        eval_retrieval_metrics(v_feat, t_feat, 'inter_class_split', args.inter_class_split_list)
    if args.intra_class_split_list is not None:
        eval_retrieval_metrics(v_feat, t_feat, 'intra_class_split', args.intra_class_split_list)

if __name__ == '__main__':
    main()