#!/bin/bash

# Check if at least two arguments are passed
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset_path> <input_json1> [<input_json2> ...]"
    exit 1
fi

# First argument is the dataset path
DATASET_PATH="$1"
shift  # Shift arguments so $@ now contains only the input_json files

# Check if the dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: dataset path '$DATASET_PATH' does not exist or is not a directory."
    exit 1
fi

# Ensure output directory exists
OUTPUT_DIR="outputs/query_poses"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Loop through all remaining arguments
for path in "$@"; do
    echo "Calling pose_estimation.py with dataset: $DATASET_PATH and input: $path"
    python tasks/pose_estimation.py \
        --dataset_path "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --input_json "$path"

    # Optional: check for error
    if [ $? -ne 0 ]; then
        echo "Error processing: $path"
    fi
done
