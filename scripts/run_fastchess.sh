#!/usr/bin/env bash
set -euo pipefail

# Example fastchess match (adjust paths/options).
# Requires fastchess in PATH and your model checkpoints.

CKPT_NEW=${1:-chess_model_transformer_weights_exp2.pth}
CKPT_OLD=${2:-chess_model_transformer_weights_exp2.pth}
OUT_PGN=${3:-out.pgn}

fastchess \
  -engine cmd="python3 uci_bot.py --ckpt ${CKPT_NEW}" name=new \
  -engine cmd="python3 uci_bot.py --ckpt ${CKPT_OLD}" name=old \
  -each tc=0.1+0.001 \
  -rounds 50 -repeat -concurrency 2 \
  -pgnout "${OUT_PGN}"
