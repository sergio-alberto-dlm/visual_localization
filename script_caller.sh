#!/bin/bash

# Check if at least one argument is passed
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path1> <path2> ..."
    exit 1
fi

# Loop through all arguments
for path in "$@"; do
    echo "Calling my_program.py with: $path"
    python tasks/rerank_retrieval.py --feats_dir outputs/local_features/ --output_dir outputs/rerank/ --input_json "$path"

    # Optional: check for error
    if [ $? -ne 0 ]; then
        echo "Error processing: $path"
    fi
done
