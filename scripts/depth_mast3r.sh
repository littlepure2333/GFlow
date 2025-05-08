#!/bin/bash

# Check if parent_dir argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 parent_dir [gpu_id] [seg_size] [scene_graph]"
    exit 1
fi

# Required argument
parent_dir="$1"

# Optional arguments with default values
gpu_id="${2:-0}"               # Default gpu_id is 0
seg_size="${3:-200}"            # Default seg_size is 200
scene_graph="${4:-logwin}"    # Default scene_graph is 'logwin'


# Execute the Python script with provided and default parameters
CUDA_VISIBLE_DEVICES=$gpu_id python utility/depth_mast3r.py \
    --parent_dir "$parent_dir" \
    --seg_size "$seg_size" \
    --scene_graph "$scene_graph"