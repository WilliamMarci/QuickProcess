#!/usr/bin/env python3
# find_failed_jobs.py

import argparse
import json
import os
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="从 metadata.json 和 status 文件中定位失败作业的 jobid"
    )
    parser.add_argument(
        "-m", "--metajson",
        required=True,
        help="metadata.json 文件路径"
    )
    parser.add_argument(
        "-s", "--status",
        required=True,
        help="status 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="将结果写入的输出文件（txt）"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 读取 metadata.json
    if not os.path.exists(args.metajson):
        print(f'Error: "{args.metajson}" not found.', file=sys.stderr)
        sys.exit(1)

    with open(args.metajson, 'r') as f:
        try:
            meta = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: 无法解析 JSON：{e}", file=sys.stderr)
            sys.exit(1)

    jobs = meta.get('jobs')
    if not isinstance(jobs, list):
        print('Error: metadata.json 格式不正确，找不到 "jobs" 列表。', file=sys.stderr)
        sys.exit(1)

    # 2. 构建 (samp, idx) -> job_list_index 的映射
    mapping = {}
    for list_idx, job in enumerate(jobs):
        samp = job.get('samp')
        idx  = job.get('idx')
        if samp is None or idx is None:
            continue
        mapping[(samp, idx)] = list_idx

    # 3. 读取 status 文件并解析失败条目
    if not os.path.exists(args.status):
        print(f'Error: "{args.status}" not found.', file=sys.stderr)
        sys.exit(1)

    pattern = re.compile(r'([^/]+)_(\d+)_tree\.root')
    failed_job_indices = set()

    with open(args.status, 'r') as f:
        for line in f:
            if 'FAILED' not in line:
                continue
            filepath = line.split(':')[0].strip()
            basename = os.path.basename(filepath)
            m = pattern.match(basename)
            if not m:
                print(f'Warning: 无法解析文件名 "{basename}"', file=sys.stderr)
                continue
            samp, idx_str = m.groups()
            idx = int(idx_str)
            key = (samp, idx)
            if key not in mapping:
                print(f'Warning: metadata 中未找到样本 "{samp}", idx={idx}', file=sys.stderr)
                continue
            failed_job_indices.add(mapping[key])

    # 4. 将结果写入输出文件
    with open(args.output, 'w') as fout:
        if not failed_job_indices:
            fout.write('No failed jobs found.\n')
        else:
            # fout.write('Failed job ids (metadata.json 中 jobs 列表的下标)：\n')
            for jid in sorted(failed_job_indices):
                fout.write(f'{jid}\n')

    print(f"Done. Results written to {args.output}")

if __name__ == '__main__':
    main()
