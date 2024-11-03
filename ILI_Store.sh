 #!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

data_paths=("ILI")
divides=("train" "val")
device="cuda:6"
num_nodes=7
input_len=36
output_lens=(60)
model_name=("gpt2")
d_model=768
l_layers=12

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    for output_len in "${output_lens[@]}"; do
      log_file="./Results/emb_time/HD_GT/mask/${data_path}_${model_name}_${output_len}_${divide}_ll12.log"
      nohup python store_hd_gt_emb.py --divide $divide --data_path $data_path --device $device --num_nodes $num_nodes --input_len $input_len --output_len $output_len --model_name $model_name --l_layers $l_layers --d_model $d_model > $log_file &
    done
  done
done