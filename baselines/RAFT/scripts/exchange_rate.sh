#!/bin/bash
# Run RAFT on exchange_rate with seq_len 96 across multiple pred_len values.
for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --model_id exchange_96_${PRED_LEN} \
    --data custom \
    --root_path ./data/exchange_rate/ \
    --data_path exchange_rate.csv \
    --seq_len 96 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --pred_len ${PRED_LEN}
done
