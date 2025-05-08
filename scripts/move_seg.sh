if [ -z "$1" ]; then
    echo "Usage: $0 parent_folder [threshold] [gpu_id]"
    exit 1
fi

parent_folder=$1
threshold=${2:-0.5}
gpu_id=${3:-0}

# Check if parent_folder exists
if [ ! -d "$parent_folder" ]; then
    echo "Error: Parent folder '$parent_folder' does not exist."
    exit 1
fi

# List all folders in the parent folder and iterate over them
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

        # Now integrate the original move_seg.py script functionality here
        echo "Running inference on img_dir: $img_dir with GPU ID: $gpu_id"
        
        # Run the move_seg Python script directly from the utility folder
        CUDA_VISIBLE_DEVICES=$gpu_id python utility/move_seg.py --img_dir $img_dir --threshold $threshold
    fi
done