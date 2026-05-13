#!/bin/bash
# Run RAFT on traffic with seq_len 96 across multiple pred_len values.
for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --model_id traffic_96_${PRED_LEN} \
    --data custom \
    --root_path ./data/traffic/ \
    --data_path traffic.csv \
    --seq_len 96 \
    --label_len 48 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --pred_len ${PRED_LEN}
done
