# export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path datasets/ \
        --data_path solar_AL.txt \
        --model_id STMTM \
        --model STMTM \
        --data Solar \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 1 \
        --enc_in 137 \
        --dec_in 137 \
        --c_out 137 \
        --d_model 32 \
        --d_ff 128 \
        --n_heads 8 \
        --kernel_size 200 \
        --seg_len 25 \
        --p_tmask 0.2 \
        --topk 3 \
        --learning_rate 0.0001 \
        --dropout 0.2 \
        --batch_size 8 \

done