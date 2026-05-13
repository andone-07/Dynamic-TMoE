python -u run.py \
    --task_name pretrain \
    --root_path datasets/ \
    --data_path exchange_rate.csv \
    --model_id STMTM \
    --model STMTM \
    --data Exchange \
    --features M \
    --seq_len 96 \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 16 \
    --d_ff 64 \
    --n_heads 16 \
    --kernel_size 100 \
    --seg_len 25 \
    --p_tmask 0.2 \
    --topk 3 \
    --learning_rate 0.001 \
    --batch_size 128 \
    --train_epochs 50 \


