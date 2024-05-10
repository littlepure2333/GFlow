cd third_party/unimatch

gpu_id=7
inference_dir="/home/wangshizun/projects/msplat/images/rollerblade/rollerblade"
output_path="/home/wangshizun/projects/msplat/images/rollerblade/rollerblade_flow"

bash scripts/gmflow_demo.sh $gpu_id $inference_dir $output_path