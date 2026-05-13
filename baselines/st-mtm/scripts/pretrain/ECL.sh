# export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path datasets/ \
    --data_path Electricity.csv \
    --model_id STMTM \
    --model STMTM \
    --data Electricity \
    --features M \
    --seq_len 96 \
    --e_layers 2 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --n_heads 4 \
    --d_model 32 \
    --d_ff 64 \
    --kernel_size 50 \
    --seg_len 25 \
    --p_tmask 0.2 \
    --topk 3 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --num_workers 0 \
    --train_epochs 10 \
