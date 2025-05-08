#!/bin/bash

# Check if parent_dir argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 sequence_path [gpu_id]"
    exit 1
fi

# Required argument
sequence_path="$1"

# Optional arguments with default values
gpu_id="${2:-0}"               # Default gpu_id is 0

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$gpu_id python gflow/fit_video.py \
--num_points 50000 \
--resize 480 \
--lr 4e-3 \
--lr_camera 0.00 \
--iterations_first 500 \
--lr_after 4e-3 \
--iterations_after 300 \
--camera_first \
--lr_camera_after 1e-3 \
--iterations_camera 150 \
--no_load_extr \
--densify_interval 150 \
--densify_times 2 \
--densify_occ_percent .5 \
--densify_interval_after 100 \
--densify_times_after 2 \
--densify_err_thre 1e-2 \
--densify_err_percent 1. \
--lambda_rgb 1. \
--lambda_depth 0.1 \
--lambda_var 50. \
--lambda_still 0. \
--lambda_flow 0.01 \
--lambda_scale 0. \
--background "black" \
--frame_range -1 \
--frame_start 0 \
--skip_interval 1 \
--traj_num 100 \
--traj_offset 2 \
--sequence_path $sequence_path \
--no_common_logs \
--logs_suffix "logs_cam_init_only"