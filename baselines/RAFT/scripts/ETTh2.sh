#!/bin/bash
# Run RAFT on ETTh2 with defaults plus fixed seq_len and multiple pred_len values.
for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --model_id ETTh2_96_${PRED_LEN} \
    --data ETTh2 \
    --root_path ./data/ETT/ \
    --data_path ETTh2.csv \
    --seq_len 96 \
    --pred_len ${PRED_LEN}
done
