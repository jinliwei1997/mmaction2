import argparse
import glob
import os
import os.path as osp
import sys
from multiprocessing import Pool


def resize_videos(full_path):
    """Generate resized video cache.

    Args:
        vid_item (list): Video item containing video full path,
            video relative path.

    Returns:
        bool: Whether generate video cache successfully.
    """
    vid_path = full_path.replace(args.src_dir,'')
    out_full_path = osp.join(args.out_dir, vid_path)
    dir_name = osp.dirname(vid_path)
    out_dir = osp.join(args.out_dir, dir_name)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    result = os.popen(
        f'ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {full_path}'  # noqa:E501
    )
    w, h = [int(d) for d in result.readline().rstrip().split(',')]
    if w > h:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i {full_path} '
               f'-vf {"mpdecimate," if args.remove_dup else ""}'
               f'scale=-2:{args.scale} '
               f'{"-vsync vfr" if args.remove_dup else ""} '
               f'-c:v libx264 {"-g 16" if args.dense else ""} '
               f'-an {out_full_path} -y')
    else:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i {full_path} '
               f'-vf {"mpdecimate," if args.remove_dup else ""}'
               f'scale={args.scale}:-2 '
               f'{"-vsync vfr" if args.remove_dup else ""} '
               f'-c:v libx264 {"-g 16" if args.dense else ""} '
               f'-an {out_full_path} -y')
    os.popen(cmd)
    print(f'{vid_path} done')
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the resized cache of original videos')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output video directory')
    parser.add_argument(
        '--dense',
        action='store_true',
        help='whether to generate a faster cache')
    parser.add_argument(
        '--level',
        type=int,
        default=4,
        help='directory level of data')
    parser.add_argument(
        '--remove-dup',
        action='store_true',
        help='whether to remove duplicated frames')
    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--scale',
        type=int,
        default=256,
        help='resize image short side length keeping ratio')
    parser.add_argument(
        '--num-worker', type=int, default=8, help='number of workers')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    lines = []
    with open('../annotation/bili_video_dm_train') as f:
        t_lines = f.readlines()
        lines.extend(t_lines)
    with open('../annotation/bili_video_dm_val') as f:
        t_lines = f.readlines()
        lines.extend(t_lines)
    print(len(lines))
    fullpath_list = [line.rstrip().split(' ##$$## ')[0] for line in lines]
    print(fullpath_list[:10])
    pool = Pool(args.num_worker)
    pool.map(resize_videos, fullpath_list)
