export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix/illness_abl" ]; then
    mkdir ./logs/FITS_fix/illness_abl
fi

model_name=FITS

for H_order in 6
do
for seq_len in 36
do
for m in 1
do
for seed in 2021
do 
for bs in 16
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id Illness_$seq_len'_'24'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 24 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/illness_abl/$m'_'$model_name'_'Illness_$seq_len'_'24'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id Illness_$seq_len'_'36'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 36 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/illness_abl/$m'_'$model_name'_'Illness_$seq_len'_'36'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id Illness_$seq_len'_'48'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 48 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/illness_abl/$m'_'$model_name'_'Illness_$seq_len'_'48'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id Illness_$seq_len'_'60'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 60 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/illness_abl/$m'_'$model_name'_'Illness_$seq_len'_'60'_H'$H_order'_bs'$bs'_s'$seed.log

done
done
done
done
done
