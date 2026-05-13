export CUDA_VISIBLE_DEVICES=0

model_name=Dynamic_TMoE

cycle_length=52
drift_window_size=520

batch_size=64
train_epochs=50
patience=15

drift_k_sigma=3.0
channel_independence=0
use_relation_layer=1
enable_drift_detection=1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 4 \
  --d_model 512 \
  --batch_size $batch_size \
  --learning_rate 0.0008 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.1 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers 2 \
  --num_rnn_layers 1 \
  --num_drift_experts 6 \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 2 \
  --d_model 1024 \
  --batch_size $batch_size \
  --learning_rate 0.0004 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.1 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers 2 \
  --num_rnn_layers 1 \
  --num_drift_experts 6 \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 4 \
  --d_model 512 \
  --batch_size $batch_size \
  --learning_rate 0.0008 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.3 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers 2 \
  --num_rnn_layers 1 \
  --num_drift_experts 6 \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_36_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 2 \
  --d_model 512 \
  --batch_size $batch_size \
  --learning_rate 0.0004 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.3 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers 1 \
  --num_rnn_layers 1 \
  --num_drift_experts 6 \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \
