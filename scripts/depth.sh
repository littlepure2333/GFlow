gpu_id=0
input_dir="/home/wangshizun/projects/gflow/data/car-turn/car-turn-test"
seg_size=200
scene_graph="swin"

cd utility/

CUDA_VISIBLE_DEVICES=$gpu_id python depth_dust3r.py \
--input_dir $input_dir \
--seg_size $seg_size \
--scene_graph $scene_graph