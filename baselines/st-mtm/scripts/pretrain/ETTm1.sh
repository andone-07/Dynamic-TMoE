# export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path datasets/ \
    --data_path ETTm1.csv \
    --model_id STMTM \
    --model STMTM \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 32 \
    --kernel_size 100 \
    --seg_len 25 \
    --p_tmask 0.2 \
    --topk 3 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --train_epochs 50 \

