# Healthy Check for ROOT Files

## `checkRootFiles`

Use `make` to build and run the `checkRootFiles` executable to verify the integrity of ROOT files in a specified directory.

Usage:
```bash
make 
./checkRootFiles <directory_path>
```

The status of each ROOT file will be output to `status.txt`.

## find_failed_job.py

This script provides an easy way to identify failed jobs by checking `status.txt` for files marked as `FAILED` and using metadata to trace back to the corresponding job directories.

Usage:
```bash
python find_failed_job.py \
    -m <path_to_metadata.json> \
    -s <path_to_status_file> \
    -o <output_result_file>
```

Parameter descriptions:
- `-m`, `--metajson`: Path to the metadata.json file
- `-s`, `--status`: Path to the status file
- `-o`, `--output`: Output result file (txt)

## tmpdisable.py

> [!WARNING]
> We recommend you not to use this script when final results are needed. This is only for temporary disabling certain functionalities during development or testing.

This script is temporarily disabled root file checking by modifying the `status.txt` marked as `FAILED`.

Usage:
```bash
./tmpdisable.py <status_txt> <log_file>
```