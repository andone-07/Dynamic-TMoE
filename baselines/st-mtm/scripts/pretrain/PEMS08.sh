python -u run.py \
    --task_name pretrain \
    --root_path datasets/PEMS08/ \
    --data_path PEMS08.npz \
    --model_id STMTM \
    --model STMTM \
    --data ETTh1 \
    --features M \
    --seq_len 336 \
    --e_layers 2 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --d_model 64 \
    --d_ff 128 \
    --n_heads 4 \
    --kernel_size 200 \
    --seg_len 25 \
    --p_tmask 0.2 \
    --topk 3 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --train_epochs 10 \


