# count=0
# while [ $count -lt 20 ]; do
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python gflow/fit_video.py \
--num_points 40000 --resize 480 \
--lr 4e-3 --lr_after 1e-3 \
--lr_camera 0.00 --lr_camera_after 1e-4 \
--camera_first --iterations_camera 150 \
--iterations_first 500 --iterations_after 300 \
--densify_interval 150 --grad_threshold 5e-3 --densify_times 2 \
--densify_interval_after 100 --grad_threshold_after 1e-4 --densify_times_after 1 \
--lambda_rgb 1. --lambda_depth 0.0001 --lambda_var 0. --lambda_still 1. --lambda_flow 0.01 \
--background "black" --frame_range 20 --frame_start 0 \
--skip_interval 1 --traj_num 100 --traj_offset 2 \
--sequence_path "./images/car-turn/car-turn"
# --sequence_path "./images/rollerblade/rollerblade"
# --sequence_path "./images/train/train"
# --sequence_path "./images/paragliding/paragliding"
# --sequence_path "/home/wangshizun/projects/gsplat/images/train/train"
# --sequence_path "/home/wangshizun/projects/gsplat/images/paragliding/paragliding"
# --sequence_path "/home/wangshizun/projects/gsplat/images/b2/b2"
# --sequence_path "/home/wangshizun/projects/gsplat/images/guitar-violin/guitar-violin"
# if [ $? -eq 0 ]; then
#     break
# fi
#     count=$((count+1))
# done

