#!/usr/bin/env python3
import os
import sys
import json
import glob
import shutil
import stat
import subprocess
import argparse
import random

################################################################################
# Basic definitions: cut function and golden JSON definitions
################################################################################
def _base_cut(choose='1L'):
    cut_dict = {
        'ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.4 && Electron_mvaFall17V2Iso_WP90',
        'mu_cut': 'Muon_pt>10 && abs(Muon_eta)<2.4 && Muon_tightId && Muon_pfRelIso04_all<0.25',
        'tight_ele_cut': 'Electron_pt>25 && Electron_mvaFall17V2Iso_WP80',
        'tight_mu_cut': 'Muon_pt>20 && Muon_pfRelIso04_all<0.15',
        'loose_ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.4 && Electron_mvaFall17V2Iso_WP90',
        'loose_mu_cut': 'Muon_pt>15 && Muon_pfRelIso04_all<0.25 && Muon_tightId',
        'jet_count': 'Sum$(Jet_pt>15 && abs(Jet_eta)<2.4 && (Jet_jetId & 4))',
        'fatjet_count': 'Sum$(FatJet_pt>150 && abs(FatJet_eta)<2.4)',
    }
    basesels = {
        '1L': '(Sum$({ele_cut} && {tight_ele_cut}) + Sum$({mu_cut} && {tight_mu_cut})) >= 1 && {jet_count} >= 3 && {fatjet_count} >= 1',
        '1L-Tight': '(Sum$({loose_ele_cut}) + Sum$({loose_mu_cut})) == 1 && (Sum$({ele_cut} && {tight_ele_cut}) + Sum$({mu_cut} && {tight_mu_cut})) >= 1 && {jet_count} >= 3 && {fatjet_count} >= 1',
    }
    return basesels[choose].format(**cut_dict)

golden_json = {
    '2015': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2016': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2017': 'Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
    '2018': 'Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt',
}

################################################################################
# Create metadata JSON file with (optional) job list; the metadata is saved in a 
# desired destination (for example the job directory) and used by processor.py.
################################################################################
def create_metadata(args, job_list=None, dest_metadata=None):
    modules = [["PhysicsTools.QuickProcess.productor.quickProducer", "quickFromConfig"]]
    metadata = {
        'cut': _base_cut(args.choose_cut),
        'branchsel_in': 'keep_and_drop_input.txt',
        'branchsel_out': 'keep_and_drop_output.txt',
        'compression': 'LZMA:9',
        'year': args.year,
        'type': args.type,
        'imports': modules
    }
    if job_list is not None:
        metadata["jobs"] = job_list
    metadata_path = dest_metadata if dest_metadata else args.metadata
    try:
        with open(metadata_path, 'w') as fp:
            json.dump(metadata, fp, indent=4)
        print(f"[INFO] Metadata saved to {metadata_path}")
    except Exception as e:
        print(f"[ERROR] Could not save metadata: {e}")

################################################################################
# Create Condor Job Directory:
#    - Place your input ROOT files into a list (optional random selection)
#    - Create a custom job directory (--jobdir)
#    - Copy the CMSSW tar archive into it (--cmssw-tar)
#    - Write metadata JSON into the job directory (including the list of jobs)
#    - Write submit.txt (job id list) and submit.cmd (Condor submit script)
#    - Copy (or write) autorun.sh and processor.py into the job directory
################################################################################
def create_job_directory(args):
    # Gather input files from --inputdir
    input_files = []
    if os.path.isdir(args.inputdir):
        for f in os.listdir(args.inputdir):
            if f.endswith('.root'):
                input_files.append(os.path.join(args.inputdir, f))
    elif os.path.isfile(args.inputdir):
        input_files.append(args.inputdir)
    else:
        raise ValueError("Input path is neither a file nor a directory")

    print(f"[INFO] Found {len(input_files)} input files.")

    # Apply random selection if --random-choice > 0
    if args.random_choice > 0:
        random.seed(42)
        input_files = random.sample(input_files, min(args.random_choice, len(input_files)))
        print(f"[INFO] Randomly selected {len(input_files)} input files.")

    # Create job directory at custom location (--jobdir)
    job_dir = os.path.abspath(args.jobdir)
    if os.path.exists(job_dir):
        print(f"[WARN] Job directory {job_dir} exists, removing it.")
        shutil.rmtree(job_dir)
    os.makedirs(job_dir)
    print(f"[INFO] Created job directory: {job_dir}")

    # Copy the CMSSW tar archive (specified by --cmssw-tar) into the job directory
    if not os.path.isfile(args.cmssw_tar):
        print(f"[ERROR] CMSSW tar archive {args.cmssw_tar} does not exist.")
        sys.exit(1)
    try:
        shutil.copy(args.cmssw_tar, os.path.join(job_dir, os.path.basename(args.cmssw_tar)))
        print(f"[INFO] Copied CMSSW tar archive to job directory.")
    except Exception as e:
        print(f"[ERROR] Failed to copy CMSSW tar archive: {e}")
        sys.exit(1)

    # Create metadata JSON (including job list) in the job directory
    dest_metadata = os.path.join(job_dir, os.path.basename(args.metadata))
    create_metadata(args, job_list=input_files, dest_metadata=dest_metadata)

    # Write submit.txt: one line per job id (0-indexed)
    submit_txt = os.path.join(job_dir, "submit.txt")
    with open(submit_txt, "w") as f:
        for i in range(len(input_files)):
            f.write(f"{i}\n")
    print(f"[INFO] Written {submit_txt} with {len(input_files)} jobs.")

    # Write submit.cmd: Condor submit script
    submit_cmd = os.path.join(job_dir, "submit.cmd")
    with open(submit_cmd, "w") as f:
        f.write(f"""Universe             = vanilla
Executable           = autorun.sh
Should_Transfer_Files = YES
WhenToTransferOutput  = ON_EXIT
transfer_input_files  = {os.path.basename(args.cmssw_tar)}, autorun.sh, processor.py, {os.path.basename(args.metadata)}, keep_and_drop_input.txt, keep_and_drop_output.txt

# Job id is provided via submit.txt
Arguments            = $(file) {args.output}

Log                  = logs/$(Cluster)_$(Process).log
Output               = logs/$(Cluster)_$(Process).out
Error                = logs/$(Cluster)_$(Process).err

Queue file from submit.txt
""")
    print(f"[INFO] Written {submit_cmd}")

    # Copy the autorun.sh and processor.py scripts into the job directory.
    # If they exist in the current directory, copy them; otherwise use built-in content.
    # Built-in content will be provided by the 'autorun_content' and 'processor_content' variables below.
    autorun_dst = os.path.join(job_dir, "autorun.sh")
    processor_dst = os.path.join(job_dir, "processor.py")

    if os.path.exists("autorun.sh"):
        shutil.copy("autorun.sh", autorun_dst)
    else:
        with open(autorun_dst, "w") as f:
            f.write(autorun_content)
    os.chmod(autorun_dst, stat.S_IRWXU)
    print(f"[INFO] Written autorun.sh in job directory.")

    if os.path.exists("processor.py"):
        shutil.copy("processor.py", processor_dst)
    else:
        with open(processor_dst, "w") as f:
            f.write(processor_content)
    os.chmod(processor_dst, stat.S_IRWXU)
    print(f"[INFO] Written processor.py in job directory.")

    # Create logs directory
    logs_dir = os.path.join(job_dir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    print(f"[SUCCESS] Job directory is ready at {job_dir}")
    print("Next steps:")
    print(f"  1) cd {job_dir} && condor_submit submit.cmd")

################################################################################
# Built-in content for autorun.sh and processor.py (if not already available)
################################################################################
autorun_content = r'''#!/bin/bash
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Arguments: $@"

# Step 1: Un-tar the CMSSW archive
CMSSW_TAR=$(ls *.tar.gz | head -1)
if [ -z "$CMSSW_TAR" ]; then
  echo "[ERROR] CMSSW tar archive not found!"
  exit 1
fi
tar -xzf "$CMSSW_TAR" --warning=no-timestamp
CMSSW_DIR=$(tar -tzf "$CMSSW_TAR" | head -1 | cut -f1 -d"/")
echo "Unpacked CMSSW directory: $CMSSW_DIR"

# Step 2: Set up CMSSW environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $CMSSW_DIR
eval $(scramv1 runtime -sh)
cd ..

# Step 3: Run processor.py with job id and output base directory
echo "[INFO] Running processor.py with arguments: $@"
./processor.py "$@"
exit $?
'''

processor_content = r'''#!/usr/bin/env python3
import sys, os, json, subprocess

# Golden JSON definitions
golden_json = {
    '2015': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2016': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2017': 'Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
    '2018': 'Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt',
}

def main():
    if len(sys.argv) != 3:
        print("Usage: processor.py <jobid> <output_base_dir>")
        sys.exit(1)
    try:
        jobid = int(sys.argv[1])
    except:
        print("Jobid must be an integer.")
        sys.exit(1)
    output_base = sys.argv[2]

    # Read metadata from metadata JSON (assumed to be in the same directory)
    metadata_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename("metadata.json"))
    try:
        with open(metadata_file) as fp:
            md = json.load(fp)
    except Exception as e:
        print(f"[ERROR] Failed to load metadata JSON: {e}")
        sys.exit(1)

    jobs = md.get("jobs", [])
    if jobid < 0 or jobid >= len(jobs):
        print(f"[ERROR] Invalid jobid: {jobid}")
        sys.exit(1)
    input_file = jobs[jobid]

    # Build module option string
    modules = []
    for mod, names in md.get("imports", []):
        modules.append(f"-I {mod} {names}")
    modules_str = " ".join(modules)

    cut = md.get("cut", "")
    branch_in = os.path.basename(md.get("branchsel_in"))
    branch_out = os.path.basename(md.get("branchsel_out"))
    compression = md.get("compression", "LZMA:9")
    firstEntry = md.get("firstEntry", 0)
    json_input = ""
    if md.get("type") == "Data":
        jf = golden_json.get(md.get("year"), "")
        json_input = f"-J ../data/JSON/{jf}"

    out_dir = os.path.join(output_base, md.get("year"), md.get("type"))
    os.makedirs(out_dir, exist_ok=True)

    cmd = (f"nano_postproc.py -c '{{cut}}' --bi {branch_in} --bo {branch_out} -z {compression} "
           f"{json_input} --first-entry {firstEntry} {out_dir} {input_file} {modules_str}")
    print(f"[INFO] Executing command: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    sys.exit(ret)

if __name__ == '__main__':
    main()
'''

################################################################################
# Argument parsing and main dispatcher
################################################################################
def get_arg_parser():
    parser = argparse.ArgumentParser(description='QuickCut: Create Condor job directory with metadata for nanoAOD processing.')
    parser.add_argument('--inputdir', '-i', required=True, help='Input nanoAOD file or directory')
    parser.add_argument('--output', '-o', required=True, help='Base output directory for processed files')
    parser.add_argument('--year', '-y', required=True, choices=['2015','2016','2017','2018'], help='Data taking year')
    parser.add_argument('--type', '-t', required=True, choices=['Data','MC'], help='Type of input files')
    parser.add_argument('--choose-cut', '-c', choices=['1L','1L-Tight'], default='1L', help='Cut configuration')
    parser.add_argument('--metadata', '-m', required=True, help='Path to metadata JSON file (will be created in job directory)')
    parser.add_argument('--jobdir', required=True, help='Path to create Condor job directory')
    parser.add_argument('--cmssw-tar', required=True, help='Path to CMSSW tar archive to copy into job directory')
    parser.add_argument('--random-choice', '-r', type=int, default=-1, help='Randomly select N input files for job creation')
    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    create_job_directory(args)