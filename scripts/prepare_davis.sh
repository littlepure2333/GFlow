#!/bin/bash

# Exit on any error
set -e

# Function to display usage
usage() {
    echo "Usage: $0 [gpu_id] [seg_size] [scene_graph] [unimatch_weight_path]"
    echo "  Optional Parameters:"
    echo "    gpu_id: GPU ID for depth and flow estimation (default 0)"
    echo "    seg_size: Segmentation size for depth estimation (default 200)"
    echo "    scene_graph: Scene graph parameter for depth estimation (default 'logwin')"
    echo "    unimatch_weight_path: Path to Unimatch weight for flow estimation (default 'third_party/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')"
    exit 1
}

# Optional Parameters with default values
gpu_id=${1:-0}
seg_size=${2:-200}
scene_graph=${3:-'logwin'}
unimatch_weight_path=${4:-'third_party/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'}

# Define paths
DOWNLOAD_PATH="./data"
UNZIP_PATH="./data/davis_ori"
TARGET_DAVIS="./data/davis"

# Step 1: Download DAVIS 2016 Dataset if not already downloaded
DAVIS_ZIP_PATH="$DOWNLOAD_PATH/DAVIS-2016.zip"
if [ ! -f "$DAVIS_ZIP_PATH" ]; then
    echo "Step 1: Downloading DAVIS 2016 dataset..."
    mkdir -p "$DOWNLOAD_PATH"
    wget -O "$DAVIS_ZIP_PATH" "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip"
    echo "DAVIS dataset downloaded."
else
    echo "DAVIS dataset already downloaded at $DAVIS_ZIP_PATH."
fi

# Step 2: Unzip the DAVIS dataset if not already unzipped
if [ ! -d "$UNZIP_PATH" ]; then
    echo "Step 2: Unzipping DAVIS dataset..."
    mkdir -p "$UNZIP_PATH"
    unzip "$DAVIS_ZIP_PATH" -d "$UNZIP_PATH"
    echo "DAVIS dataset unzipped to $UNZIP_PATH."
else
    echo "DAVIS dataset already unzipped at $UNZIP_PATH."
fi

# Step 3: Organize DAVIS Dataset Structure
echo "Step 3: Organizing DAVIS dataset..."
./scripts/organize_davis.sh "$UNZIP_PATH/DAVIS/JPEGImages/480p" "$TARGET_DAVIS"
echo "Dataset organized at $TARGET_DAVIS."

# Step 4: Estimate Depth using MASt3R
echo "Step 4: Estimating depth..."
./scripts/depth_mast3r.sh "$TARGET_DAVIS" "$gpu_id" "$seg_size" "$scene_graph"
echo "Depth estimation completed."

# Step 5: Estimate Flow using Unimatch
echo "Step 5: Estimating flow..."
./scripts/flow_unimatch.sh "$TARGET_DAVIS" "$unimatch_weight_path" "$gpu_id"
echo "Flow estimation completed."

echo "Step 6: Estimating segmentation..."
./scripts/move_seg.sh "$TARGET_DAVIS" 0.5 "$gpu_id"
echo "Segmentation estimation completed."

echo "Dataset preparation complete!"