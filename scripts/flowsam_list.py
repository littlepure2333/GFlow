import os
import subprocess

# list all folders in the "data/tapvid-davis/images" 
folder_list = sorted(os.listdir("data/tapvid-davis/images"))
print(folder_list)
# exit()

for folder in folder_list:
    img_dir = os.path.join("/home/wangshizun/projects/gflow/data/tapvid-davis/images", folder, folder)
    save_path = f"{img_dir}_flowsam"

    # command = (
    #     "cd third_party/flowsam && "
    #     f"python evaluation.py --model=flowpsam --flow_gaps=1,-1,2,-2 "
    #     f"--max_obj=1 --num_gridside=20 "
    #     f"--dataset=gflow --img_dir={img_dir} "
    #     f"--ckpt_path='checkpoints/frame_level_flowpsam_vitbvith_train_on_oclrsyn_dvs17.pth' "
    #     f"--save_path={save_path}"
    # )

    command = "pwd"
    command = "bash scripts/flowsam.sh " + img_dir

    print("Running command on img_dir:", img_dir)
    subprocess.run(command, shell=True)