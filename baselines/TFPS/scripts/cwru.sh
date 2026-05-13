if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/cwru" ]; then
    mkdir ./logs/LongForecasting/cwru
fi

model_name=PatchTST_MoE_cluster

root_path_name=./dataset/CWRU/
data_path_name=CWRU_Time_Normal_1_098_ds10.csv
model_id_name=CWRU
data_name=CWRU

batch_size=128

for seq_len in 96
do
for pred_len in 96 192 336 720
do
for random_seed in 2021
do
for learning_rate in 0.0005
do
for T_num_expert in 4
do
for T_top_k in 1
do
for F_num_expert in 4
do
for F_top_k in 1
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --target X098_DE_time \
      --freq s \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 2 \
      --dec_in 2 \
      --c_out 2 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --individual 0\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 50\
      --num_workers 0 \
      --gpu 0\
      --itr 1 --batch_size $batch_size --learning_rate $learning_rate >logs/LongForecasting/cwru/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$T_num_expert'_'$T_top_k'_'$F_num_expert'_'$F_top_k'_'$learning_rate'_0.1'.log
done
done
done
done
done
done
done
done
