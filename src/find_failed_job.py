#!/usr/bin/env python3
# find_failed_jobs.py

import argparse
import json
import os
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate failed job IDs from metadata.json and status file"
    )
    parser.add_argument(
        "-m", "--metajson",
        required=True,
        help="Path to metadata.json file"
    )
    parser.add_argument(
        "-s", "--status",
        required=True,
        help="Path to status file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file (txt) to write the results"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Read metadata.json
    if not os.path.exists(args.metajson):
        print(f'Error: "{args.metajson}" not found.', file=sys.stderr)
        sys.exit(1)

    with open(args.metajson, 'r') as f:
        try:
            meta = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON: {e}", file=sys.stderr)
            sys.exit(1)

    jobs = meta.get('jobs')
    if not isinstance(jobs, list):
        print('Error: Invalid metadata.json format, "jobs" list not found.', file=sys.stderr)
        sys.exit(1)

    # 2. Build (samp, idx) -> job_list_index mapping
    mapping = {}
    for list_idx, job in enumerate(jobs):
        samp = job.get('samp')
        idx  = job.get('idx')
        if samp is None or idx is None:
            continue
        mapping[(samp, idx)] = list_idx

    # 3. Read status file and parse failed entries
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
                print(f'Warning: Failed to parse filename "{basename}"', file=sys.stderr)
                continue
            samp, idx_str = m.groups()
            idx = int(idx_str)
            key = (samp, idx)
            if key not in mapping:
                print(f'Warning: Sample "{samp}", idx={idx} not found in metadata.', file=sys.stderr)
                continue
            failed_job_indices.add(mapping[key])

    # 4. Write results to output file
    with open(args.output, 'w') as fout:
        if not failed_job_indices:
            fout.write('No failed jobs found.\n')
        else:
            for jid in sorted(failed_job_indices):
                fout.write(f'{jid}\n')

    print(f"Done. Results written to {args.output}")

if __name__ == '__main__':
    main()
