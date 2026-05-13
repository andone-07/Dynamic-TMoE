python -u run.py \
    --task_name pretrain \
    --root_path datasets/ \
    --data_path solar_AL.txt \
    --model_id STMTM \
    --model STMTM \
    --data Solar \
    --features M \
    --seq_len 336 \
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
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 10 \


