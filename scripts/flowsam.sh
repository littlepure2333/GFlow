# img_dir="../../data/car-turn/car-turn"
img_dir=$1
save_path=$img_dir"_flowsam"

cd third_party/flowsam

python evaluation.py --model=flowpsam --flow_gaps=1,-1,2,-2 \
                      --max_obj=1 --num_gridside=20 \
                      --dataset=gflow --img_dir=$img_dir \
                      --ckpt_path="checkpoints/frame_level_flowpsam_vitbvith_train_on_oclrsyn_dvs17.pth" \
                      --save_path=$save_path