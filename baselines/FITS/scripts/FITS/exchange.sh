export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix/exchange_abl" ]; then
    mkdir ./logs/FITS_fix/exchange_abl
fi

model_name=FITS

for H_order in 6 5 4 3
do
for seq_len in 720 360 180 90
do
for m in 1 2
do
for seed in 114
do 
for bs in 32
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'96'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 8 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/exchange_abl/$m'_'$model_name'_'Exchange_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'192'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/exchange_abl/$m'_'$model_name'_'Exchange_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'336'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 8 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/exchange_abl/$m'_'$model_name'_'Exchange_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'720'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 8 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/exchange_abl/$m'_'$model_name'_'Exchange_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log

done
done
done
done
done
