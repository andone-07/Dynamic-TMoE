#!/bin/bash
# Run RAFT on weather with seq_len 96 across multiple pred_len values.
for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --model_id weather_96_${PRED_LEN} \
    --data custom \
    --root_path ./data/weather/ \
    --data_path weather.csv \
    --seq_len 96 \
    --label_len 48 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --pred_len ${PRED_LEN}
done
