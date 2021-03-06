import numpy as np
import argparse
import pickle
def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('--v_t_result', help='V_T feature pickle file path')
    parser.add_argument('--label_result', default=None, help='label result')
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
    cls_cnt = [0 for i in range(212)]
    cls_cnt_pos = [0 for i in range(212)]
    for i in range(N):
        cls_cnt[label[i]] += 1
        if gt_rank[i] == 0:
            cls_cnt_pos[label[i]] += 1

    cls_acc = [cls_cnt_pos[i]/cls_cnt[i] for i in range(212)]
    mca = sum(cls_acc)/212
    mean_rk, median_rk, recall1, recall5, recall10 = cal_avg_metrics(np.array(gt_rank))
    print('---V to T Zero Shot Classification---\n')
    print(f'mca {mca}')
    print(f'vt_mean_rk\t{mean_rk:.4f}\nvt_median_rk\t{median_rk:.4f}\nvt_recall1\t{recall1:.4f}\nvt_recall5\t{recall5:.4f}\nvt_recall10\t{recall10:.4f}\n')


def main():
    args = parse_args()
    anno = '/mnt/lustre/jinliwei/annotation/usv_val_list_frame_text_label'
    lines = open(anno, 'r').readlines()
    result = pickle.load(open(args.label_result, 'rb'))

    d = {}
    for i in range(len(result)):
        d[lines[i].rstrip().split(' ')[0].split('/')[-2]] = result[i][1][0]

    results = pickle.load(open(args.v_t_result, 'rb'))
    v_feat = np.array([result[0] for result in results])
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