#!/bin/bash
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Arguments: $@"

# Step 1: Un-tar the CMSSW archive (assumes one *.tar.gz in the current directory)
CMSSW_TAR=$(ls *.tar.gz | head -1)
if [ -z "$CMSSW_TAR" ]; then
  echo "[ERROR] CMSSW tar archive not found!"
  exit 1
fi
tar -xzf "$CMSSW_TAR" --warning=no-timestamp
CMSSW_DIR=$(tar -tzf "$CMSSW_TAR" | head -1 | cut -f1 -d"/")
echo "Unpacked CMSSW directory: $CMSSW_DIR"

# Step 2: Set up the CMSSW environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd $CMSSW_DIR
eval $(scramv1 runtime -sh)
cd ..

# Step 3: Run processor.py with job id and output base directory
echo "[INFO] Running processor.py with arguments: $@"
./processor.py "$@"
exit $?