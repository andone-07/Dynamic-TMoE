export CUDA_VISIBLE_DEVICES=0

model_name=Dynamic_TMoE

drift_window_size=5760
num_temporal_moe_layers=2
num_drift_experts=3

batch_size=256
train_epochs=60
patience=15
cycle_length=24
drift_k_sigma=3.0
channel_independence=0
use_relation_layer=1

enable_drift_detection=1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 6 \
  --batch_size $batch_size \
  --d_model 128 \
  --learning_rate 0.0008 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.8 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers $num_temporal_moe_layers \
  --num_rnn_layers 1 \
  --num_drift_experts $num_drift_experts \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 6 \
  --batch_size $batch_size \
  --d_model 256 \
  --learning_rate 0.0004 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.6 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers $num_temporal_moe_layers \
  --num_rnn_layers 1 \
  --num_drift_experts $num_drift_experts \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 6 \
  --batch_size $batch_size \
  --d_model 256 \
  --learning_rate 0.0004 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.6 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers $num_temporal_moe_layers \
  --num_rnn_layers 1 \
  --num_drift_experts $num_drift_experts \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patch_len 24 \
  --stride 6 \
  --batch_size $batch_size \
  --d_model 128 \
  --learning_rate 0.0004 \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout 0.8 \
  --cycle_length $cycle_length \
  --drift_window_size $drift_window_size \
  --drift_k_sigma $drift_k_sigma \
  --num_temporal_moe_layers $num_temporal_moe_layers \
  --num_rnn_layers 1 \
  --num_drift_experts $num_drift_experts \
  --channel_independence $channel_independence \
  --use_relation_layer $use_relation_layer \
  --enable_drift_detection $enable_drift_detection \