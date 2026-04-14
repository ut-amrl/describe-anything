#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://localhost:9014}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-2}"

usage() {
    cat <<'EOF'
Usage: smoke_test_describe_anything_api.sh [options]

Checks DAM API liveness by POSTing an empty body to /chat/completions.
A 422 response is treated as success (route + request validation reached).
A 200 response is also treated as success.

Options:
  --url <url>                  Server URL (default: http://localhost:9014)
  --timeout-seconds <seconds>  Max wait time (default: 240)
  --interval-seconds <seconds> Retry interval (default: 2)
  -h, --help                   Show this help
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --url) SERVER_URL="$2"; shift ;;
        --timeout-seconds) TIMEOUT_SECONDS="$2"; shift ;;
        --interval-seconds) INTERVAL_SECONDS="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown parameter: $1"; usage; exit 1 ;;
    esac
    shift
done

ENDPOINT="${SERVER_URL%/}/chat/completions"
DEADLINE=$((SECONDS + TIMEOUT_SECONDS))

while (( SECONDS < DEADLINE )); do
    HTTP_CODE="$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$ENDPOINT" -H "Content-Type: application/json" -d '{}' || true)"

    if [[ "$HTTP_CODE" == "422" || "$HTTP_CODE" == "200" ]]; then
        echo "DAM API smoke test passed at $ENDPOINT (HTTP $HTTP_CODE)"
        exit 0
    fi

    echo "Waiting for DAM API at $ENDPOINT (last HTTP code: ${HTTP_CODE:-none})"
    sleep "$INTERVAL_SECONDS"
done

echo "ERROR: DAM API smoke test timed out after ${TIMEOUT_SECONDS}s: $ENDPOINT"
exit 1
