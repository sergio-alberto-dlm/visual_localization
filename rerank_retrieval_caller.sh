#!/bin/bash

# Check if at least two arguments are passed
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <feats_dir> <input_json1> [<input_json2> ...]"
    exit 1
fi

# First argument is the features directory
FEATS_DIR="$1"
shift  # Shift arguments so $@ now contains only the input_json files

# Check if the features directory exists
if [ ! -d "$FEATS_DIR" ]; then
    echo "Error: features directory '$FEATS_DIR' does not exist or is not a directory."
    exit 1
fi

# Ensure output directory exists
OUTPUT_DIR="outputs/rerank"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Loop through all remaining arguments (JSON inputs)
for path in "$@"; do
    echo "Calling rerank_retrieval.py with feats: $FEATS_DIR and input: $path"
    python tasks/rerank_retrieval.py \
        --feats_dir "$FEATS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --input_json "$path"

    # Optional: check for error
    if [ $? -ne 0 ]; then
        echo "Error processing: $path"
    fi
done
