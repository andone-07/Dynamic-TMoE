#!/bin/bash
# Run RAFT on CWRU bearing data with seq_len 96 across multiple pred_len values.
# Dataset split follows Dataset_Custom in code: train/val/test = 70%/10%/20% (time-ordered).

LOG_DIR="./logs/CWRU"
mkdir -p "${LOG_DIR}"
RUN_TS=$(date +"%Y%m%d_%H%M%S")

for PRED_LEN in 96 192 336 720; do
  LOG_FILE="${LOG_DIR}/CWRU_sl96_pl${PRED_LEN}_${RUN_TS}.log"
  echo "[CWRU] Start run: pred_len=${PRED_LEN}, log=${LOG_FILE}"

  python -u run.py \
    --is_training 1 \
    --model_id CWRU_96_${PRED_LEN} \
    --data CWRU \
    --root_path ./data/CWRU/ \
    --data_path CWRU_Time_Normal_1_098_ds10.csv \
    --features M \
    --target X098_FE_time \
    --freq s \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${PRED_LEN} \
    --enc_in 2 \
    --dec_in 2 \
    --c_out 2 \
    --gpu 1 2>&1 | tee "${LOG_FILE}"

done
