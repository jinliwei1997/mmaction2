import numpy as np
import argparse
import pickle
def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('--v_t_feature', help='V_T feature pickle file path')
    parser.add_argument('--label_feature_dict', default=None, help='label_feature_dict')
    parser.add_argument('--anno', default='/mnt/lustre/jinliwei/annotation/usv_val_list_frame_text_title', help='anno')
    args = parser.parse_args()
    return args


def cal_avg_metrics(gt_rank):
    N = gt_rank.shape[0]
    mean_rk = np.mean(gt_rank)
    median_rk = np.median(gt_rank)
    recall1 = np.sum(gt_rank < 1) / N
    recall5 = np.sum(gt_rank < 5) / N
    recall10 = np.sum(gt_rank < 10) / N
    return mean_rk, median_rk, recall1, recall5, recall10

def eval_zero_shot(v_feat, label_feat, label):
    N = v_feat.shape[0]
    s = np.matmul(v_feat, np.transpose(label_feat))
    rank = np.argsort(s)[:, ::-1]
    mask = np.repeat(label.reshape(N, 1), axis=1, repeats=212)
    gt_rank = np.argsort(rank == mask)[:, ::-1][:, :1].reshape(N)

    mean_rk, median_rk, recall1, recall5, recall10 = cal_avg_metrics(np.array(gt_rank))
    print('---V to T Zero Shot Classification---\n')
    print(f'vt_mean_rk\t{mean_rk:.4f}\nvt_median_rk\t{median_rk:.4f}\nvt_recall1\t{recall1:.4f}\nvt_recall5\t{recall5:.4f}\nvt_recall10\t{recall10:.4f}\n')


def main():
    args = parse_args()
    results = pickle.load(open(args.v_t_feature, 'rb'))
    v_feat = np.array([result[0] for result in results])
    d = pickle.load(open(args.label_feature_dict, 'rb'))
    label_feat = []
    label_dict = {}
    cnt = 0
    for key in d:
        label_feat.append(d[key])
        label_dict[key] = cnt
        cnt += 1

    lines = open(args.anno, 'r').readlines()
    label = [line.rstrip().split(' ')[0].split('/')[-2] for line in lines]
    label_id = [label_dict[i] for i in label]

    eval_zero_shot(v_feat, np.array(label_feat), np.array(label_id))

if __name__ == '__main__':
    main()