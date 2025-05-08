import os
import glob
from benchmark import main
import argparse


'''gflow'''
# Parse command line arguments
parser = argparse.ArgumentParser(description="Benchmark multiple sequences.")
parser.add_argument("--path", type=str, required=True, help="Path to the data directory.")
parser.add_argument("--log_suffix", type=str, default="logs_cam_init_only", help="Suffix for the log directory.")
args = parser.parse_args()

path = args.path
log_suffix = args.log_suffix


# using glob to get all the folders in the path
folders = glob.glob(os.path.join(path, "*"))
folders.sort()  # sort the folders
csv = {}

for folder in folders:
    folder_name = os.path.basename(folder)
    print(f"Evaluating {folder_name}...")
    sequence_path = os.path.join(folder, folder_name)
    log_path_latest = os.path.join(folder, f"{folder_name}_{log_suffix}/0_latest")
    if os.path.exists(log_path_latest):
        # the subfolder in the log_path
        subfolders = glob.glob(os.path.join(log_path_latest, "*"))
        if len(subfolders) > 0:
            log_path = subfolders[0]
            if os.path.isdir(log_path):
                print(sequence_path)
                print(log_path)
                csv_a = main(log_path=log_path, sequence_path=sequence_path, csv_name=log_suffix, eval_recon=True, eval_track=True, eval_seg=True, eval_camera=True)
                csv[folder_name] = csv_a

# Get the headers
headers = list(csv[list(csv.keys())[0]].keys())

# Init average results dic
avg = {}
avg_validate_counts = {}
for header in headers:
    avg[header] = 0
    avg_validate_counts[header] = 0

# Save the csv into a file
csv_path = os.path.join(path, "metrics.csv")
with open(csv_path, "w") as f:
    # Write the headers
    f.write("sequence,")
    for header in headers:
        f.write(f"{header},")
    f.write("\n")
    # Write the data
    for key in csv.keys():
        f.write(f"{key},")
        for header in headers:
            value = csv[key][header]
            f.write(f"{value},")
            if value:
                avg[header] += value
                avg_validate_counts[header] += 1
        f.write("\n")
    # Write the average
    f.write("Average,")
    for header in headers:
        avg[header] /= avg_validate_counts[header]
        f.write(f"{avg[header]},")
    f.close()
print(f"Metrics saved in {csv_path}")