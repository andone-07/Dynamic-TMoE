#!/bin/bash
# Run RAFT on MIT-BIH ECG data with seq_len 96 across multiple pred_len values.
# Dataset split follows Dataset_Custom in code: train/val/test = 70%/10%/20% (time-ordered).

LOG_DIR="./logs/MIT-BIH"
mkdir -p "${LOG_DIR}"
RUN_TS=$(date +"%Y%m%d_%H%M%S")

for PRED_LEN in 96 192 336 720; do
  LOG_FILE="${LOG_DIR}/MIT-BIH_sl96_pl${PRED_LEN}_${RUN_TS}.log"
  echo "[MIT-BIH] Start run: pred_len=${PRED_LEN}, log=${LOG_FILE}"

  python -u run.py \
    --is_training 1 \
    --model_id MIT-BIH_96_${PRED_LEN} \
    --data custom \
    --root_path ./data/MIT-BIH/ \
    --data_path 16265_ds10.csv \
    --features M \
    --target lead_1 \
    --freq s \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${PRED_LEN} \
    --enc_in 2 \
    --dec_in 2 \
    --c_out 2 2>&1 | tee "${LOG_FILE}"

done
