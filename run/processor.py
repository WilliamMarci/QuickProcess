#!/usr/bin/env python3
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

    # Read metadata from metadata.json (assumed to be in same directory)
    metadata_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.json")
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

    # Build modules option string
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