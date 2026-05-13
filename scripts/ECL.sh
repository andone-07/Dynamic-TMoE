export CUDA_VISIBLE_DEVICES=0

model_name=Dynamic_TMoE

drift_window_size=2400
num_drift_experts=3
batch_size=32
train_epochs=50
patience=10
cycle_length=24
drift_k_sigma=3.0
channel_independence=0
use_relation_layer=1
enable_drift_detection=1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 6 \
  --batch_size $batch_size \
  --d_model 128 \
  --learning_rate 0.0008 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.1 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers 2 \
  --num_rnn_layers 2 \
  --num_drift_experts $num_drift_experts \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 6 \
  --batch_size $batch_size \
  --d_model 128 \
  --learning_rate 0.0008 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.3 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers 2 \
  --num_rnn_layers 2 \
  --num_drift_experts $num_drift_experts \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 6 \
  --batch_size $batch_size \
  --d_model 256 \
  --learning_rate 0.0012 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.3 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers 2 \
  --num_rnn_layers 1 \
  --num_drift_experts $num_drift_experts \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \
  
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 6 \
  --batch_size $batch_size \
  --d_model 128 \
  --learning_rate 0.0008 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.3 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers 2 \
  --num_rnn_layers 2 \
  --num_drift_experts $num_drift_experts \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \