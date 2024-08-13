import os
import subprocess

# list all folders in the "data/tapvid-davis/images" 
folder_list = sorted(os.listdir("data/tapvid-davis/images"))
print(folder_list)
# exit()

for folder in folder_list:
    # transfer_dir = os.path.join("/home/wangshizun/projects/gflow/data/tapvid-davis/images", folder, folder+"_flow_refine")
    transfer_dir = os.path.join("/home/wangshizun/projects/gflow/data/tapvid-davis/images", folder, folder+"_epipolar")
    

    # command = (
    #     "cd third_party/flowsam && "
    #     f"python evaluation.py --model=flowpsam --flow_gaps=1,-1,2,-2 "
    #     f"--max_obj=1 --num_gridside=20 "
    #     f"--dataset=gflow --img_dir={img_dir} "
    #     f"--ckpt_path='checkpoints/frame_level_flowpsam_vitbvith_train_on_oclrsyn_dvs17.pth' "
    #     f"--save_path={save_path}"
    # )

    command = "rsync -avhP -r " + transfer_dir + f" wangshizun@10.246.112.240:/home/wangshizun/projects/msplat/data_share/tapvis_davis/images/{folder}/"
    print("Running command on transfer_dir:", transfer_dir)
    subprocess.run(command, shell=True)