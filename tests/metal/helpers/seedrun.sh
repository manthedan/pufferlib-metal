#!/bin/bash
# usage: seedrun.sh BINDIR ENV SEED [extra args]  -> prints "score sha16 sps"
set -euo pipefail
D=$1; ENV=$2; SEED=$3; shift 3
cd "$D" || exit 1
OUT=$(./puffer train --base.seed=$SEED "$@" 2>&1)
INI=$(ls -t "logs/$ENV/"*.ini | head -1)
ID=$(basename "$INI" .ini)
SCORE=$(grep '^env/score' "$INI" | tail -1 | sed 's/.*= *//' | tr ',' '\n' | tail -1)
CK=$(ls "checkpoints/$ENV/$ID/"*.bin 2>/dev/null | sort | tail -1)
SHA=$(shasum -a 256 "$CK" | cut -c1-16)
SPS=$(echo "$OUT" | grep -o 'SPS[^0-9]*[0-9.]*[KM]\?' | tail -1)
echo "$D seed=$SEED score=$SCORE sha=$SHA $SPS"
