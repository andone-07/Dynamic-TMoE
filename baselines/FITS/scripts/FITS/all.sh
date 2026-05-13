export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix" ]; then
    mkdir ./logs/FITS_fix
fi

if [ ! -d "./logs/FITS_fix/etth1_abl" ]; then
    mkdir ./logs/FITS_fix/etth1_abl
fi

model_name=FITS

for H_order in 6
do
for seq_len in 96
do
for m in 1
do
for seed in 2021
do 
for bs in 64 #256 #32 64 # 128 256
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth1_$seq_len'_'96'_H'$H_order'_s'$seed"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth1_$seq_len'_'192'_H'$H_order'_s'$seed"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth1_$seq_len'_'336'_H'$H_order'_s'$seed"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth1_$seq_len'_'720'_H'$H_order'_s'$seed"

done
done
done
done
done

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix" ]; then
    mkdir ./logs/FITS_fix
fi

if [ ! -d "./logs/FITS_fix/etth2_abl" ]; then
    mkdir ./logs/FITS_fix/etth2_abl
fi
# seq_len=700
model_name=FITS

for H_order in 6
do
for seq_len in 96
do
for m in 1
do
for seed in 2021
do 
for bs in 64 #256 #32 64 # 128 256
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth2_abl/$m'_'$model_name'_'Etth2_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth2_$seq_len'_'192'_H'$H_order'_s'$seed"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth2_abl/$m'_'$model_name'_'Etth2_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth2_$seq_len'_'336'_H'$H_order'_s'$seed"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth2_abl/$m'_'$model_name'_'Etth2_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth2_$seq_len'_'720'_H'$H_order'_s'$seed"

done
done
done
done
done

for H_order in 6
do
for seq_len in 96
do
for m in 2
do
for seed in 2021
do 
for bs in 64 #256 #32 64 # 128 256
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth2_abl/$m'_'$model_name'_'Etth2_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $model_name'_'Etth2_$seq_len'_'96'_H'$H_order'_s'$seed"

done
done
done
done
done

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix" ]; then
    mkdir ./logs/FITS_fix
fi

if [ ! -d "./logs/FITS_fix/ettm1_abl" ]; then
    mkdir ./logs/FITS_fix/ettm1_abl
fi

model_name=FITS

for H_order in 14
do
for seq_len in 96
do
for m in 1
do
for seed in 2021
do
for bs in 64 # 128 256
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm1_abl/$m'_'$model_name'_'Ettm1_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm1_abl/$m'_'$model_name'_'Ettm1_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $m'_'$model_name'_'Ettm1_$seq_len'_'192'_H'$H_order"

done
done
done
done
done

for H_order in 14
do
for seq_len in 96
do
for m in 1
do
for seed in 2021
do
for bs in 64 # 128 256
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm1_abl/$m'_'$model_name'_'Ettm1_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $m'_'$model_name'_'Ettm1_$seq_len'_'336'_H'$H_order"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm1_abl/$m'_'$model_name'_'Ettm1_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log

  echo "Done $m'_'$model_name'_'Ettm1_$seq_len'_'720'_H'$H_order"

done
done
done
done
done

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix" ]; then
    mkdir ./logs/FITS_fix
fi

if [ ! -d "./logs/FITS_fix/ettm2_abl" ]; then
    mkdir ./logs/FITS_fix/ettm2_abl
fi

model_name=FITS

for H_order in 14
do
for seq_len in 96
do
for m in 2
do
for seed in 2021
do
for bs in 64 # 128 256
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20\
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm2_abl/$m'_'$model_name'_'Ettm2_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20\
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm2_abl/$m'_'$model_name'_'Ettm2_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20\
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm2_abl/$m'_'$model_name'_'Ettm2_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --base_T 96 \
  --gpu 0 \
  --seed $seed \
  --patience 20\
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/ettm2_abl/$m'_'$model_name'_'Ettm2_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log

done
done
done
done
done


if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix" ]; then
    mkdir ./logs/FITS_fix
fi

if [ ! -d "./logs/FITS_fix/exchange_abl" ]; then
    mkdir ./logs/FITS_fix/exchange_abl
fi

model_name=FITS

for H_order in 6
do
for seq_len in 96
do
for m in 1
do
for seed in 2021
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

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix" ]; then
    mkdir ./logs/FITS_fix
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


if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix" ]; then
    mkdir ./logs/FITS_fix
fi

if [ ! -d "./logs/FITS_fix/traf_abl" ]; then
    mkdir ./logs/FITS_fix/traf_abl
fi

model_name=FITS

for H_order in 10
do
for seq_len in 96
do
for m in 1
do
for seed in 2021
do
for bs in 64
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_j'96'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/traf_abl/$m'j_'$model_name'_'Traffic_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log

  # echo "Done with $m'j_'$model_name'_'Traffic_$seq_len'_'96'_H'$H_order'_s'$seed.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_j'192'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/traf_abl/$m'j_'$model_name'_'Traffic_$seq_len'_'192'_H'$H_order'_bs'$bs'_s'$seed.log

  # echo "Done with $m'j_'$model_name'_'Traffic_$seq_len'_'192'_H'$H_order'_s'$seed.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_j'336'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/traf_abl/$m'j_'$model_name'_'Traffic_$seq_len'_'336'_H'$H_order'_bs'$bs'_s'$seed.log

  # echo "Done with $m'j_'$model_name'_'Traffic_$seq_len'_'336'_H'$H_order'_s'$seed.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id Traffic_$seq_len'_j'720'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 862 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 10 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/traf_abl/$m'j_'$model_name'_'Traffic_$seq_len'_'720'_H'$H_order'_bs'$bs'_s'$seed.log

  # echo "Done with $m'j_'$model_name'_'Traffic_$seq_len'_'720'_H'$H_order'_s'$seed.log"

done
done
done
done
done


if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/FITS_fix" ]; then
    mkdir ./logs/FITS_fix
fi

if [ ! -d "./logs/FITS_fix/weather_abl1" ]; then
    mkdir ./logs/FITS_fix/weather_abl1
fi

model_name=FITS

for seq_len in 96
do
for bs in 32
do
for seed in 2021
do

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id Weather_$seq_len'_j'96'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --train_mode 2 \
  --H_order 12 \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual --seed $seed | tee logs/FITS_fix/weather_abl1/$m'j_'$model_name'_'Weather_$seq_len'_'96'_H'$H_order'_bs'$bs'_s_'$seed'.log' 

  # echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'96'_H'$H_order.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id Weather_$seq_len'_j'192'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 21 \
  --des 'Exp' \
  --train_mode 1 \
  --H_order 12 \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual --seed $seed | tee logs/FITS_fix/weather_abl1/$m'j_'$model_name'_'Weather_$seq_len'_'192'_H'$H_order'_bs'$bs'_s_'$seed'.log' 

  # echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'192'_H'$H_order.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id Weather_$seq_len'_j'336'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 21 \
  --des 'Exp' \
  --train_mode 2 \
  --H_order 8 \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual --seed $seed | tee logs/FITS_fix/weather_abl1/$m'j_'$model_name'_'Weather_$seq_len'_'336'_H'$H_order'_bs'$bs'_s_'$seed'.log' 

  # echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'336'_H'$H_order.log"

python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id Weather_$seq_len'_j'720'_H'$H_order \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 21 \
  --des 'Exp' \
  --train_mode 1 \
  --H_order 12 \
  --base_T 144 \
  --gpu 0 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 --individual --seed $seed | tee logs/FITS_fix/weather_abl1/$m'j_'$model_name'_'Weather_$seq_len'_'720'_H'$H_order'_bs'$bs'_s_'$seed'.log' 

  # echo "Done with $m'j_'$model_name'_'Weather_$seq_len'_'720'_H'$H_order.log"


done
done
done
