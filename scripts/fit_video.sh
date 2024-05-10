# count=0
# while [ $count -lt 20 ]; do
CUDA_VISIBLE_DEVICES=7 python gflow/fit_video.py \
--num_points 20000 --resize 480 \
--lr 4e-3 --lr_after 1e-3 \
--lr_camera 0.00 --lr_camera_after 1e-4 \
--camera_first --iterations_camera 100 \
--iterations_first 1000 --iterations_after 800 \
--densify_interval 200 --grad_threshold 5e-3 --densify_times 2 \
--densify_interval_after 5 --grad_threshold_after 1e-4 --densify_times_after 1 \
--lambda_depth 0.01 --lambda_var 0. --lambda_still 0.001 --lambda_flow 0.01 \
--depth_scale 1.0 --depth_offset 1. \
--background "black" --frame_range 20 --frame_start 20 \
--skip_interval 1 --no_add \
--sequence_path "./images/car-turn/car-turn"
# --sequence_path "./images/hike/hike"
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

