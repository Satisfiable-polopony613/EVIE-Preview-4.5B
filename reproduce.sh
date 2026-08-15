#!/usr/bin/env bash
# Reproduce the ViDoRe V1/V2/V3 numbers on every visible GPU.
#
#   bash reproduce.sh [/path/to/vidore]
#
# The dataset path may also come from VIDORE_DATA, or from a local.env file
# next to this script holding PYTHON= and VIDORE_DATA=.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f local.env ]] && source local.env

PYTHON="${PYTHON:-python3}"
DATA="${1:-${VIDORE_DATA:-./vidore}}"

if [[ ! -d "$DATA/eval" ]]; then
  echo "ViDoRe data not found at: $DATA" >&2
  echo "usage: bash reproduce.sh /path/to/vidore" >&2
  exit 1
fi

GPUS="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
[[ "$GPUS" -ge 1 ]] || { echo "no GPU visible" >&2; exit 1; }

LOG="reproduce_$(date +%Y%m%d_%H%M%S).log"
nohup "$PYTHON" -m torch.distributed.run --nproc_per_node="$GPUS" \
  reproduce.py --data-root "$DATA" --output results.json >"$LOG" 2>&1 &

echo "running on $GPUS GPU(s), pid $!"
echo "tail -f $(pwd)/$LOG"
