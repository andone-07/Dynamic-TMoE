export CUDA_VISIBLE_DEVICES=1

python -u run.py \
    --task_name pretrain \
    --root_path datasets/ \
    --data_path MIT-BIH/16265_ds10.csv \
    --model_id STMTM \
    --model STMTM \
    --data MIT-BIH \
    --features M \
    --target lead_1 \
    --freq s \
    --seq_len 96 \
    --e_layers 2 \
    --enc_in 2 \
    --dec_in 2 \
    --c_out 2 \
    --n_heads 2 \
    --d_model 16 \
    --d_ff 64 \
    --kernel_size 25 \
    --seg_len 25 \
    --p_tmask 0.2 \
    --topk 1 \
    --learning_rate 0.001 \
    --batch_size 128 \
    --num_workers 0 \
    --train_epochs 50 \
