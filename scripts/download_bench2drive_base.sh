#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data/bench2drive_base"
RAW_DIR="${DATA_DIR}/raw"
JSON_FILE="${PROJECT_ROOT}/../simlingo/Bench2Drive/docs/bench2drive_base_1000.json"

if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found at $JSON_FILE"
    exit 1
fi

mkdir -p "${RAW_DIR}"
mkdir -p "${DATA_DIR}/extracted"

if ! command -v huggingface-cli &> /dev/null; then
    pip install huggingface-hub
fi

if [ ! -f "${DATA_DIR}/file_list.txt" ]; then
    python3 -c "
import json
with open('${JSON_FILE}', 'r') as f:
    data = json.load(f)
with open('${DATA_DIR}/file_list.txt', 'w') as f:
    for filename in data.keys():
        f.write(filename + '\n')
"
fi

TOTAL_FILES=$(wc -l < "${DATA_DIR}/file_list.txt")
DOWNLOADED=$(find "${RAW_DIR}" -name "*.tar.gz" 2>/dev/null | wc -l)

echo "Total: ${TOTAL_FILES}, Downloaded: ${DOWNLOADED}, Remaining: $((TOTAL_FILES - DOWNLOADED))"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

COUNTER=1
while IFS= read -r filename; do
    if [ -f "${RAW_DIR}/${filename}" ]; then
        echo "[${COUNTER}/${TOTAL_FILES}] Skip: ${filename}"
        COUNTER=$((COUNTER + 1))
        continue
    fi
    
    echo "[${COUNTER}/${TOTAL_FILES}] Downloading: ${filename}"
    huggingface-cli download --repo-type dataset rethinklab/Bench2Drive \
        --include "${filename}" \
        --local-dir "${RAW_DIR}" \
        --local-dir-use-symlinks False || echo "Failed: ${filename}"
    
    COUNTER=$((COUNTER + 1))
    
    if [ $((COUNTER % 50)) -eq 0 ]; then
        DOWNLOADED_NOW=$(find "${RAW_DIR}" -name "*.tar.gz" 2>/dev/null | wc -l)
        echo "Progress: ${DOWNLOADED_NOW}/${TOTAL_FILES}"
    fi
done < "${DATA_DIR}/file_list.txt"

echo "Done. Files in: ${RAW_DIR}"
