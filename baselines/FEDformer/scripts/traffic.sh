export CUDA_VISIBLE_DEVICES=1

for preLen in 96 192 336 720
do

python -u run.py \
 --is_training 1 \
 --root_path ./dataset/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model FEDformer \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 1 \
 --train_epochs 3

done