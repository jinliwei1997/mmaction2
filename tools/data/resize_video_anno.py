import argparse
import glob
import os
import os.path as osp
import sys
from multiprocessing import Pool
from pathlib import Path

def resize_videos(full_path, time_step = 10):
    """Generate resized video cache.

    Args:
        vid_item (list): Video item containing video full path,
            video relative path.

    Returns:
        bool: Whether generate video cache successfully.
    """
    vid_path = full_path.replace(args.src_dir,'')
    partition = vid_path.split('/')[0]
    vid_name = vid_path.split('/')[-1]
    BV = vid_path.split('/')[-2]
    out_full_path = osp.join(args.out_dir, partition, BV, vid_name)
    out_dir = osp.join(args.out_dir, partition, BV)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    result = os.popen(
        f'ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "{full_path}"'  # noqa:E501
    )
    print("------------sb1-----------")
    sys.stdout.flush()
    w, h = [int(d) for d in result.readline().rstrip().split(',')]
    if w > h:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i "{full_path}" '
               f'-vf {"mpdecimate," if args.remove_dup else ""}'
               f'scale=-2:{args.scale} '
               f'{"-vsync vfr" if args.remove_dup else ""} '
               f'-c:v libx264 {"-g 16" if args.dense else ""} '
               f'-an "{out_full_path}" -y')
    else:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i "{full_path}" '
               f'-vf {"mpdecimate," if args.remove_dup else ""}'
               f'scale={args.scale}:-2 '
               f'{"-vsync vfr" if args.remove_dup else ""} '
               f'-c:v libx264 {"-g 16" if args.dense else ""} '
               f'-an "{out_full_path}" -y')

    r = os.popen(cmd)
    r.readlines()
    print("------------sb-----------")
    sys.stdout.flush()
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{out_full_path}"'
    r = os.popen(cmd)
    print(r.readlines())
    sys.stdout.flush()
    duration = int(float(r.readlines().rstrip()))

    print(duration)
    sys.stdout.flush()
    for i in range(0, duration, time_step):
        cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{out_full_path}"'
        r = os.popen(cmd)
        path_i = out_full_path[:-4]+f'_&_{i}.mp4'
        r.readlines(f'ffmpeg -i {out_full_path} -ss {i} -t {time_step} {path_i}')

    print(f'{out_full_path} done; time: {duration}s')
    sys.stdout.flush()
    return True

def gao(full_path):
    try:
        resize_videos(full_path)
    except:
        print(f'error {full_path}')

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
    pool.map(gao, fullpath_list)
