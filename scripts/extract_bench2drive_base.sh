#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data/bench2drive_base"
RAW_DIR="${DATA_DIR}/raw"
EXTRACT_DIR="${DATA_DIR}/extracted"

if [ ! -d "${RAW_DIR}" ]; then
    echo "Error: ${RAW_DIR} not found"
    exit 1
fi

TAR_COUNT=$(find "${RAW_DIR}" -name "*.tar.gz" | wc -l)
if [ ${TAR_COUNT} -eq 0 ]; then
    echo "Error: No tar.gz files found"
    exit 1
fi

echo "Found ${TAR_COUNT} files to extract"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

COUNTER=1
for tarfile in "${RAW_DIR}"/*.tar.gz; do
    if [ -f "$tarfile" ]; then
        filename=$(basename "$tarfile")

        # Peek inside to derive a marker for whether we've already extracted it.
        # Uses the first path entry from the tarball; falls back to filename.
        first_entry=$(tar -tzf "$tarfile" 2>/dev/null | head -1)
        first_entry=${first_entry#./}          # drop leading "./" if present
        marker=${first_entry%%/*}
        marker=${marker:-$filename}

        if [ -e "${EXTRACT_DIR}/${marker}" ]; then
            echo "[${COUNTER}/${TAR_COUNT}] Skip (already extracted): $filename"
        else
            echo "[${COUNTER}/${TAR_COUNT}] Extracting: $filename"
            tar -xzf "$tarfile" -C "${EXTRACT_DIR}"
        fi

        COUNTER=$((COUNTER + 1))
        
        if [ $((COUNTER % 100)) -eq 0 ]; then
            echo "Progress: ${COUNTER}/${TAR_COUNT}"
        fi
    fi
done

echo "Done. Extracted to: ${EXTRACT_DIR}"
