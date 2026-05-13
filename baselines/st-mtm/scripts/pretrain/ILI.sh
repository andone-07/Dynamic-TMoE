# export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path datasets/ \
    --data_path national_illness.csv \
    --model_id STMTM \
    --model STMTM \
    --data ILI \
    --features M \
    --seq_len 36 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 32 \
    --d_ff 32 \
    --kernel_size 25 \
    --seg_len 3 \
    --p_tmask 0.6 \
    --topk 4 \
    --learning_rate 0.001 \
    --batch_size 8 \
    --train_epochs 50 \

