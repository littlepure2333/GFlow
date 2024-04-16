# count=0
# while [ $count -lt 20 ]; do
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=5 python gflow/fit_video.py \
--num_points 20000 --lr 4e-3 --lr_after 1e-3 \
--iterations_first 1000 --iterations_after 500 \
--densify_interval 200 --grad_threshold 5e-3 --densify_times 2 \
--densify_interval_after 50 --grad_threshold_after 1e-6 --densify_times_after 1 \
--lambda_depth 1e-3 --lambda_flow 1e-4 \
--depth_scale 2.0 \
--background "black" --frame_range -1 --frame_start 0 \
--skip_interval 1 --no_add --loss_verbose --resize 480 \
--sequence_path "/home/wangshizun/projects/gsplat/images/car-turn/car-turn"
# --sequence_path "/home/wangshizun/projects/gsplat/images/train/train"
# --sequence_path "/home/wangshizun/projects/gsplat/images/paragliding/paragliding"
# --sequence_path "/home/wangshizun/projects/gsplat/images/b2/b2"
# --sequence_path "/home/wangshizun/projects/gsplat/images/guitar-violin/guitar-violin"
# if [ $? -eq 0 ]; then
#     break
# fi
#     count=$((count+1))
# done

