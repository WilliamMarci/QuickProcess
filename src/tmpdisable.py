#!/usr/bin/env python3
import os
import sys

def process_files(status_txt, log_file):
    with open(status_txt, 'r') as f, open(log_file, 'w') as log:
        for line in f:
            line = line.strip()
            if line.endswith('FAILED') or line.endswith('Failed'):
                filepath = line.split(' : ')[0]
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    bak_path = filepath + '.bak'
                    os.rename(filepath, bak_path)
                    log.write(f'{bak_path}\t{size}\n')
                    print(f'Renamed: {filepath} -> {bak_path}, Size: {size}')
                else:
                    print(f'File not found: {filepath}')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <status_txt> <log_file>")
        sys.exit(1)
    process_files(sys.argv[1], sys.argv[2])