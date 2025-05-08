#!/bin/bash

# Check if both arguments (source_folder and target_folder) are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 source_folder target_folder"
    exit 1
fi

# Assign the first and second arguments to variables
source_folder="$1"
target_folder="$2"

# Check if source_folder exists
if [ ! -d "$source_folder" ]; then
    echo "Error: Source folder '$source_folder' does not exist."
    exit 1
fi

# Iterate through each scene in the source folder
for scene in "$source_folder"/*; do
    if [ -d "$scene" ]; then
        # Get the base name of the scene folder
        scene_name=$(basename "$scene")

        # Create the target directory structure
        target_scene_folder="$target_folder/$scene_name/$scene_name"
        mkdir -p "$target_scene_folder"

        # Copy all images from the source scene to the target scene folder
        cp "$scene"/* "$target_scene_folder"

        echo "Copied images from $scene to $target_scene_folder"
    fi
done

echo "All scenes processed successfully."