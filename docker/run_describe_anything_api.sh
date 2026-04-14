#!/usr/bin/env bash
set -euo pipefail

PORT="${DAM_API_PORT:-9014}"
HOST="${DAM_API_HOST:-0.0.0.0}"
APP_MODULE="${DAM_APP_MODULE:-dam_server:app}"
PROJECT_ROOT="${DAM_PROJECT_ROOT:-/workspace/project}"

cd "$PROJECT_ROOT"

# ------------------------------------------------------------
# GPU selection logic (priority order):
# 1. CUDA_VISIBLE_DEVICES (user explicitly set)
# 2. DAM_API_CUDA_VISIBLE_DEVICES (container-specific fallback)
# 3. otherwise: leave as-is (all GPUs visible)
# ------------------------------------------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Using pre-set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
elif [[ -n "${DAM_API_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${DAM_API_CUDA_VISIBLE_DEVICES}"
  echo "Using DAM_API_CUDA_VISIBLE_DEVICES -> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
  echo "CUDA_VISIBLE_DEVICES not set (all visible GPUs available to API)"
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Starting DAM API"
echo "  project root: $PROJECT_ROOT"
echo "  app module:   $APP_MODULE"
echo "  host:         $HOST"
echo "  port:         $PORT"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

exec uvicorn "$APP_MODULE" \
  --host "$HOST" \
  --port "$PORT" \
  --log-level info