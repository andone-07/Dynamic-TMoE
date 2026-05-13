# export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path datasets/ \
    --data_path Weather.csv \
    --model_id STMTM \
    --model STMTM \
    --data Weather \
    --features M \
    --seq_len 96 \
    --e_layers 2 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --kernel_size 100 \
    --seg_len 25 \
    --p_tmask 0.2 \
    --topk 3 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --train_epochs 50 \

