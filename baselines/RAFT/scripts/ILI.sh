#!/bin/bash
# Run RAFT on ILI with seq_len 36 across specified pred_len values.
for PRED_LEN in 24 36 48 60; do
  python -u run.py \
    --model_id ILI_36_${PRED_LEN} \
    --data custom \
    --root_path ./data/illness/ \
    --data_path national_illness.csv \
    --label_len 18 \
    --seq_len 36 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --pred_len ${PRED_LEN}
done
