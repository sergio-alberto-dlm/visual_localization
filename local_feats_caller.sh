#!/bin/bash

# Check if exactly three arguments are passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset_path> <output_path> <building>"
    echo "  building must be one of: HYDRO, SUCCULENT, BOTH"
    exit 1
fi

# Assign arguments
DATASET_PATH="$1"
OUTPUT_PATH="$2"
BUILDING="$3"

# Check dataset path
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: dataset path '$DATASET_PATH' does not exist or is not a directory."
    exit 1
fi

# Ensure output path exists
if [ ! -d "$OUTPUT_PATH" ]; then
    echo "Creating output directory: $OUTPUT_PATH"
    mkdir -p "$OUTPUT_PATH"
fi

# Validate building argument
if [[ "$BUILDING" != "HYDRO" && "$BUILDING" != "SUCCULENT" && "$BUILDING" != "BOTH" ]]; then
    echo "Error: invalid building '$BUILDING'. Must be HYDRO, SUCCULENT, or BOTH."
    exit 1
fi

# If BOTH, expand into both buildings
if [ "$BUILDING" == "BOTH" ]; then
    BUILDINGS=("HYDRO" "SUCCULENT")
else
    BUILDINGS=("$BUILDING")
fi

# Loop through selected buildings
for b in "${BUILDINGS[@]}"; do
    echo "Calling local_feats.py with dataset: $DATASET_PATH, output: $OUTPUT_PATH, building: $b"
    python tasks/local_feats.py \
        --dataset_path "$DATASET_PATH" \
        --output_path "$OUTPUT_PATH" \
        --building "$b"

    # Optional: check for error
    if [ $? -ne 0 ]; then
        echo "Error processing building: $b"
    fi
done
