#!/usr/bin/env bash
#
# Download dataset from https://osf.io/rymks/ which includes data for the
# experiments shown in `examples/nhp_mapping.json` and `examples/rat_mapping.json`
#
# Usage:
#   ./download_nhp_rat_dataset.sh <OUTPUT_PATH>
#
# NHP data will be found under `<OUTPUT_PATH>/nhp` and
# Rat data will be found under `<OUTPUT_PATH>/rat`

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input param for output path
OUTPUT_PATH=$1

# Get starting time
start=`date +%s`

# SCRIPT STARTS HERE
# ==============================================================================
# Create output folders for NHP and rat if applicable
mkdir -p ${OUTPUT_PATH}/rat ${OUTPUT_PATH}/nhp

# Download .mat data files for NHP
curl -L -o ${OUTPUT_PATH}/nhp/Cebus1_M1_190221.mat https://osf.io/rymks/download
curl -L -o ${OUTPUT_PATH}/nhp/Cebus2_M1_200123.mat https://osf.io/abzkg/download
curl -L -o ${OUTPUT_PATH}/nhp/Macaque1_M1_181212.mat https://osf.io/bnv8c/download
curl -L -o ${OUTPUT_PATH}/nhp/Macaque2_M1_190527.mat https://osf.io/c3dvb/download

# Download .mat data files for rat
curl -L -o ${OUTPUT_PATH}/rat/rat1_M1_190716.mat https://osf.io/kwdg3/download
curl -L -o ${OUTPUT_PATH}/rat/rat2_M1_190617.mat https://osf.io/wg9sb/download
curl -L -o ${OUTPUT_PATH}/rat/rat3_M1_190728.mat https://osf.io/pfnb6/download
curl -L -o ${OUTPUT_PATH}/rat/rat4_M1_191109.mat https://osf.io/rjfne/download
curl -L -o ${OUTPUT_PATH}/rat/rat5_M1_191112.mat https://osf.io/sfvnj/download
curl -L -o ${OUTPUT_PATH}/rat/rat6_M1_200218.mat https://osf.io/wh4a3/download

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "Downloaded rat and NHP data successfully!"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"