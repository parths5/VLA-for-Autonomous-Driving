#!/bin/bash
set -e

DATA_DIR="/ocean/projects/cis250252p/shared/VLAD/data/bench2drive_mini"
mkdir -p "${DATA_DIR}/raw"
mkdir -p "${DATA_DIR}/extracted"

if ! command -v huggingface-cli &> /dev/null; then
    pip install huggingface-hub
fi

cd "${DATA_DIR}/raw"

FILES=(
    "HardBreakRoute_Town01_Route30_Weather3.tar.gz"
    "DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz"
    "Accident_Town03_Route156_Weather0.tar.gz"
    "YieldToEmergencyVehicle_Town04_Route165_Weather7.tar.gz"
    "ConstructionObstacle_Town05_Route68_Weather8.tar.gz"
    "ParkedObstacle_Town10HD_Route371_Weather7.tar.gz"
    "ControlLoss_Town11_Route401_Weather11.tar.gz"
    "AccidentTwoWays_Town12_Route1444_Weather0.tar.gz"
    "OppositeVehicleTakingPriority_Town13_Route600_Weather2.tar.gz"
    "VehicleTurningRoute_Town15_Route443_Weather1.tar.gz"
)

for i in "${!FILES[@]}"; do
    file="${FILES[$i]}"
    echo "[$((i+1))/10] $file"
    huggingface-cli download --repo-type dataset rethinklab/Bench2Drive \
        --include "$file" \
        --local-dir . \
        --local-dir-use-symlinks False
done

echo "Done. Files in: ${DATA_DIR}/raw"
