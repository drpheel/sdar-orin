#!/usr/bin/env bash
# Resume the GSM8K run. Finished problems stay in eval_gsm8k/results.jsonl.
# Do not pass --overwrite; that deletes the saved grades.
cd "$(dirname "$0")"
exec python3 eval_gsm8k.py \
  --limit 100 \
  --steps 32 \
  --max-new-tokens 512 \
  --out eval_gsm8k
