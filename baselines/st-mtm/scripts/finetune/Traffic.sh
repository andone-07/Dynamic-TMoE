# export CUDA_VISIBLE_DEVICES=0

for pred_len in 96; do
    python -u run.py \
        --task_name finetune \
        --is_training 0 \
        --root_path datasets/ \
        --data_path traffic.csv \
        --model_id STMTM \
        --model STMTM \
        --data Traffic \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --n_heads 4 \
        --d_model 32 \
        --d_ff 64 \
        --dropout 0.2 \
        --batch_size 128 \
        --learning_rate 0.0001 \
        --kernel_size 50 \
        --seg_len 25 \
        --p_tmask 0.2 \
        --topk 3 \

done

