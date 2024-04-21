CUDA_VISIBLE_DEVICES=6 python gflow/fit_video_match.py \
--depth_scale 10.0 --resize 480 --traj_num 200 \
--sequence_path "./images/car-turn/car-turn" \
--load_path "./logs/2024_04_21-09_13_05"
