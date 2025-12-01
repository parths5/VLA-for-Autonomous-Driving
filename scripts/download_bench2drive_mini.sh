#!/bin/bash
# Download Bench2Drive Mini Dataset (10 clips, ~4GB)

set -e  # Exit on error

# Create directories
DATA_DIR="/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_mini"
mkdir -p "${DATA_DIR}/raw"
mkdir -p "${DATA_DIR}/extracted"

echo "========================================="
echo "Downloading Bench2Drive Mini Dataset"
echo "Target directory: ${DATA_DIR}"
echo "========================================="

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Installing..."
    pip install huggingface-hub
fi

# Download the 10 mini dataset clips
cd "${DATA_DIR}/raw"

echo ""
echo "Downloading clip 1/10: HardBreakRoute_Town01..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "HardBreakRoute_Town01_Route30_Weather3.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "Downloading clip 2/10: DynamicObjectCrossing_Town02..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "Downloading clip 3/10: Accident_Town03..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "Accident_Town03_Route156_Weather0.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "Downloading clip 4/10: YieldToEmergencyVehicle_Town04..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "YieldToEmergencyVehicle_Town04_Route165_Weather7.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "Downloading clip 5/10: ConstructionObstacle_Town05..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "ConstructionObstacle_Town05_Route68_Weather8.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "Downloading clip 6/10: ParkedObstacle_Town10HD..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "ParkedObstacle_Town10HD_Route371_Weather7.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "Downloading clip 7/10: ControlLoss_Town11..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "ControlLoss_Town11_Route401_Weather11.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "Downloading clip 8/10: AccidentTwoWays_Town12..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "AccidentTwoWays_Town12_Route1444_Weather0.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "Downloading clip 9/10: OppositeVehicleTakingPriority_Town13..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "OppositeVehicleTakingPriority_Town13_Route600_Weather2.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "Downloading clip 10/10: VehicleTurningRoute_Town15..."
huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive \
    --include "VehicleTurningRoute_Town15_Route443_Weather1.tar.gz" \
    --local-dir . --local-dir-use-symlinks False

echo ""
echo "========================================="
echo "Download complete!"
echo "Files saved to: ${DATA_DIR}/raw/"
echo ""
echo "To extract the data, run:"
echo "  bash scripts/extract_bench2drive_mini.sh"
echo "========================================="



