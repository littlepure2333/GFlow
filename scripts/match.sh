CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 python gflow/fit_video_match.py \
--no_traj_only --no_fixed_camera --depth_scale 2.0 --resize 480 \
--sequence_path "/home/wangshizun/projects/msplat/images/car-turn/car-turn" \
--load_path "/home/wangshizun/projects/msplat/logs/2024_04_15-04_01_47"
