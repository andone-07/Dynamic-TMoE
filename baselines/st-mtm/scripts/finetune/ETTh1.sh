# export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path datasets/ \
        --data_path ETTh1.csv \
        --model_id STMTM \
        --model STMTM \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --e_layers 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 16 \
        --d_ff 64 \
        --n_heads 8 \
        --kernel_size 200 \
        --seg_len 25 \
        --p_tmask 0.2 \
        --topk 3 \
        --learning_rate 0.0001 \
        --dropout 0.2 \
        --batch_size 128 \

done