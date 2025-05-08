#!/bin/bash

# Read the parent folder from the command-line argument
if [ -z "$1" ]; then
    echo "Usage: $0 parent_folder [resume_weight_path] [gpu_id]"
    exit 1
fi

# Define the parent folder
parent_folder=$1


# Optional resume weight path with a default value
resume=${2:-"third_party/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"}

# Optional GPU ID with a default value of 0 if not provided
gpu_id=${3:-0}

# Check if parent_folder exists
if [ ! -d "$parent_folder" ]; then
    echo "Error: Parent folder '$parent_folder' does not exist."
    exit 1
fi

# Add third_party/unimatch to the system PATH so any scripts or binaries can be executed without specifying the full path
export PATH=$PATH:$(pwd)/third_party/unimatch

# Iterate over each folder in the parent folder
for folder in "$parent_folder"/*; do
    if [ -d "$folder" ]; then
        # Extract the base name of the folder
        folder_name=$(basename "$folder")
        img_dir="$folder/$folder_name"

        # Check if the img_dir exists before proceeding
        if [ ! -d "$img_dir" ]; then
            echo "Warning: img_dir '$img_dir' does not exist, skipping."
            continue
        fi

        # Now integrate the original flow.sh script functionality here
        echo "Running inference on img_dir: $img_dir with GPU ID: $gpu_id"
        
        # Set output path for this particular scene
        output_path="${img_dir}_flow_unimatch"

        # Run the flow refinement Python script directly from the third_party/unimatch folder
        CUDA_VISIBLE_DEVICES=$gpu_id python third_party/unimatch/main_flow.py \
            --inference_dir "$img_dir" \
            --resume "$resume" \
            --output_path "$output_path" \
            --padding_factor 32 \
            --upsample_factor 4 \
            --num_scales 2 \
            --attn_splits_list 2 8 \
            --corr_radius_list -1 4 \
            --prop_radius_list -1 1 \
            --reg_refine \
            --num_reg_refine 6 \
            --pred_bidir_flow \
            --fwd_bwd_check \
            --save_flo_flow

        echo "Finished processing $img_dir"
    fi
done

echo "All processes completed successfully."