#!/bin/bash
# Run RAFT on electricity with seq_len 96 across multiple pred_len values.
for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --model_id electricity_96_${PRED_LEN} \
    --data custom \
    --root_path ./data/electricity/ \
    --data_path electricity.csv \
    --seq_len 96 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --pred_len ${PRED_LEN}
done
