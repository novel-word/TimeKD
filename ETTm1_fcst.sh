#!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

seq_lens=(96)
pred_lens=(96)
learning_rates=(1e-4)
channels=(512)
d_llm=(768)
e_layers=(6)
dropout_ns=(0.1)
batch_sizes=(16)

model_name="gpt2"
data_path="ETTm1"
epochs=(100)
recon_w=(0.5)
feature_w=(0.01)
att_w=(0.01)

for seq_len in "${seq_lens[@]}"; do 
  for pred_len in "${pred_lens[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for channel in "${channels[@]}"; do
        for dropout_n in "${dropout_ns[@]}"; do
          for e_layer in "${e_layers[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
              log_path="./Results/Fcst/${data_path}/HD_GT/ll12/"
              mkdir -p $log_path
              log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dn${dropout_n}_bs${batch_size}_e${epochs}_rw${recon_w}_fw${feature_w}_aw${att_w}.log"
              nohup python train_fcst_off_fd.py \
                --data_path $data_path \
                --device cuda:6 \
                --batch_size $batch_size \
                --num_nodes 7 \
                --seq_len $seq_len \
                --pred_len $pred_len \
                --epochs $epochs \
                --seed 6666 \
                --channel $channel \
                --head 4 \
                --lrate $learning_rate \
                --dropout_n $dropout_n \
                --e_layer $e_layer\
                --model_name $model_name \
                --feature_w $feature_w \
                --fcst_w 1 \
                --recon_w $recon_w \
                --att_w $att_w\
                --num_workers 10 \
                --d_llm $d_llm > $log_file &
            done
          done
        done
      done
    done
  done
done