# export CUDA_VISIBLE_DEVICES=0

for pred_len in 24 36 48 60; do
    python -u run.py \
        --task_name finetune \
        --is_training 0 \
        --root_path datasets/ \
        --data_path national_illness.csv \
        --model_id STMTM \
        --model STMTM \
        --data ILI \
        --features M \
        --seq_len 36 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 32 \
        --d_ff 32 \
        --dropout 0.2 \
        --batch_size 128 \
        --learning_rate 0.0001 \
        --kernel_size 25 \
        --seg_len 3 \
        --p_tmask 0.2 \
        --topk 3 \

done

