gpu_id=1
# inference_dir="/home/wangshizun/projects/gflow/data/car-turn/car-turn"
inference_dir=$1
output_path=$inference_dir"_flow_refine"

cd third_party/unimatch

# CUDA_VISIBLE_DEVICES=$gpu_id python main_flow.py \
# --inference_dir $inference_dir \
# --resume pretrained/gmflow-scale2-mixdata-train320x576-9ff1c094.pth \
# --output_path $output_path \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --pred_bidir_flow \
# --fwd_bwd_check \
# --save_flo_flow


CUDA_VISIBLE_DEVICES=$gpu_id python main_flow.py \
--inference_dir $inference_dir \
--resume pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
--output_path $output_path \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6 \
--pred_bidir_flow \
--fwd_bwd_check \
--save_flo_flow