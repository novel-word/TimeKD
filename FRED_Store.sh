 #!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

data_paths=("FRED")
divides=("train" "val" "test")
device="cuda:7"
num_nodes=107
input_len=36
output_len=24
model_name=("gpt2")
d_model=768

for data_path in "${data_paths[@]}"; do
  for divide in "${divides[@]}"; do
    log_file="./Results/emb_time/GT_LT/${data_path}_${divide}.log"
    nohup python store_gt_lt_emb.py --divide $divide --data_path $data_path --device $device --num_nodes $num_nodes --input_len $input_len --output_len $output_len --model_name $model_name --d_model $d_model > $log_file &
  done
done