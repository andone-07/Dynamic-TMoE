#!/bin/bash
# Run RAFT on ETTm1 with defaults plus fixed seq_len and multiple pred_len values.
for PRED_LEN in 96 192 336 720; do
  python -u run.py \
    --model_id ETTm1_96_${PRED_LEN} \
    --data ETTm1 \
    --root_path ./data/ETT/ \
    --data_path ETTm1.csv \
    --seq_len 96 \
    --pred_len ${PRED_LEN}
done
