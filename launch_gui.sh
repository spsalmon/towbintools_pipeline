#!/usr/bin/env bash

# --- Configuration: edit these ---
FILEMAP_PATH="/mnt/towbin.data/shared/aslesarchuk/20260421_squid5_10x_160_186_415_393_272_634/analysis/report/analysis_filemap.csv"
OPEN_ANNOTATED=1      # 1 = open annotated if exists, 0 = always open original
RECOMPUTE_VALUES_AT_MOLT=0            # 1 = recompute features at molt, 0 = skip if already computed
PORT=0             # 0 = random available port
HOST="127.0.0.1"
# ---------------------------------

~/.local/bin/micromamba run -n towbintools python "./gui/run.py" \
    ${FILEMAP_PATH:+--filemap "$FILEMAP_PATH"} \
    ${RECOMPUTE_VALUES_AT_MOLT:+$([ "$RECOMPUTE_VALUES_AT_MOLT" = "1" ] && echo "--recompute")} \
    $([ "$OPEN_ANNOTATED" = "0" ] && echo "--no-annotated") \
    --host "$HOST" \
    --port "$PORT"
