#!/usr/bin/env bash

# --- Configuration: edit these ---
FILEMAP_PATH="/mnt/towbin.data/shared/spsalmon/pipeline_test_folder/analysis/report/analysis_filemap.parquet"
OPEN_ANNOTATED=1       # 1 = open annotated if exists, 0 = always open original
RECOMPUTE=0            # 1 = recompute features at molt, 0 = skip if already computed
PORT=0             # 0 = random available port
HOST="127.0.0.1"
# ---------------------------------

~/.local/bin/micromamba run -n towbintools python "./gui/run.py" \
    ${FILEMAP_PATH:+--filemap "$FILEMAP_PATH"} \
    ${RECOMPUTE:+$([ "$RECOMPUTE" = "1" ] && echo "--recompute")} \
    $([ "$OPEN_ANNOTATED" = "0" ] && echo "--no-annotated") \
    --host "$HOST" \
    --port "$PORT"
