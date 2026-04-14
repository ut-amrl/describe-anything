#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-describe_anything_container}"
IMAGE_TAG="${IMAGE_TAG:-describe-anything:latest}"
DAM_MODEL_PATH="${DAM_MODEL_PATH:-nvidia/DAM-3B}"
REBUILD=0
RESTART=0
FORCE_FROM_SCRATCH=0
USE_TIMESTAMP_TAG=0
DETACH=0
SKIP_LOCAL_BUILD=0
START_API=0
AUTO_SMOKE_TEST=1
SMOKE_TIMEOUT_SECONDS="${DAM_API_SMOKE_TIMEOUT_SECONDS:-240}"
SMOKE_INTERVAL_SECONDS="${DAM_API_SMOKE_INTERVAL_SECONDS:-2}"

PRIMARY_REMOTE_HOST="${PRIMARY_REMOTE_HOST:-robolidar}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-$PWD}"

HOST_PORT="${HOST_PORT:-9014}"
CONTAINER_PORT="${CONTAINER_PORT:-9014}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-}"
CONTAINER_CHECKPOINTS_DIR="${CONTAINER_CHECKPOINTS_DIR:-/workspace/checkpoints}"
AUTO_DOWNLOAD_IF_MISSING="${AUTO_DOWNLOAD_IF_MISSING:-true}"

# Host path bind-mounted as HF_HOME inside the container (persistent Hub cache).
# Default: /scratch/aaadkins/hf_cache. Override with --hf-home or DAM_HF_HOME_HOST.
# To skip the mount, export DAM_HF_HOME_HOST= (empty) before running; --hf-home always wins.
HF_HOME_HOST="${DAM_HF_HOME_HOST-/scratch/aaadkins/hf_cache}"
CONTAINER_HF_HOME="${CONTAINER_HF_HOME:-/workspace/hf_home}"
DATA_DIR_HOST="${DATA_DIR_HOST:-/robodata/aaadkins}"
CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR:-/workspace/data}"

# UI/X11 support is enabled by default. Use --no-ui to disable.
ENABLE_UI=1
DISPLAY_VALUE="${DISPLAY:-:0}"
XSOCK_DIR="${XSOCK_DIR:-/tmp/.X11-unix}"
XAUTHORITY_HOST="${XAUTHORITY_HOST:-${XAUTHORITY:-}}"

usage() {
    cat <<'EOF'
Usage: run_describe_anything_container.sh [options]

Without --api: idle container (sleep); host port is still mapped so you can start the API later inside the container.
With --api: runs uvicorn for the DAM service on container start.

Options:
  --api                          Start the HTTP API inside the container (uvicorn)
  --rebuild                      Rebuild image, remove old container, and run fresh
  --restart                      Recreate existing container without forcing image rebuild
  -f, --force                    Rebuild with --no-cache
  --timestamp-tag                Build with timestamp tag (also retags latest)
  --no-local-build               Skip local build and use remote fallback flow
  --remote-host <host>           Primary remote build host
  --remote-project-dir <path>    Remote repo path for build context
    --image-tag <tag>              Image tag to run (default: describe-anything:latest)
  --container-name <name>        Container name
  --host-port <port>             Host port mapped to the container API port (default: 9014; always published)
    --container-port <port>        Container port for the API when you run run_describe_anything_api.sh (default: 9014)
  --checkpoints-dir <path>       Host checkpoints directory to bind mount
  --container-checkpoints-dir    Container checkpoints path (default: /workspace/checkpoints)
  --auto-download <true|false>   Download checkpoint via Python API when missing
  --hf-home <path>               Host directory for Hugging Face Hub cache (default: /scratch/aaadkins/hf_cache)
  --data-dir <path>              Host data directory to bind mount (default: /robodata/aaadkins)
  --container-data-dir <path>    Container data path (default: /workspace/data)
    --skip-smoke-test              Skip automatic API smoke test in --api mode
    --smoke-timeout <seconds>      API smoke test timeout (default: 240)
    --smoke-interval <seconds>     API smoke test retry interval (default: 2)

UI/X11 options (enabled by default):
  --no-ui                        Disable GUI/X11 support
  --display <display>            Display to use for GUI apps (default: current DISPLAY or :0)
  --xauthority <path>            Host Xauthority file to mount into container

General:
  --detach                       Do not open shell after startup
  -h, --help                     Show this help

Environment (optional):
    DAM_HF_HOME_HOST=<path>        Hub cache host dir (default /scratch/aaadkins/hf_cache if unset)
    DAM_HF_HOME_HOST=              Empty string: do not bind-mount HF_HOME (ephemeral cache in container)
  HF_TOKEN                       If set on the host, passed into the container (not printed)
    DAM_DEVICE                     e.g. cpu or cuda — passed into the container when set
    DAM_PODMAN_NVIDIA_MODE         auto (default), cdi, or legacy — see README (Podman 3 vs 4 GPU passthrough)
  XSOCK_DIR                      X11 socket dir on host (default: /tmp/.X11-unix)
  XAUTHORITY_HOST                Xauthority file on host (default: $XAUTHORITY if set)

GPU subset for the API only (container still sees all GPUs for other work):
    DAM_API_CUDA_VISIBLE_DEVICES=0,1 ./docker/run_describe_anything_container.sh --api --detach

Passed into the container as DAM_API_CUDA_VISIBLE_DEVICES; run_describe_anything_api.sh applies it
only when starting uvicorn. Rebuild not required.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rebuild) REBUILD=1 ;;
        --restart) RESTART=1 ;;
        -f|--force) FORCE_FROM_SCRATCH=1 ;;
        --timestamp-tag) USE_TIMESTAMP_TAG=1 ;;
        --no-local-build) SKIP_LOCAL_BUILD=1 ;;
        --remote-host) PRIMARY_REMOTE_HOST="$2"; shift ;;
        --remote-project-dir) REMOTE_PROJECT_DIR="$2"; shift ;;
        --image-tag) IMAGE_TAG="$2"; shift ;;
        --container-name) CONTAINER_NAME="$2"; shift ;;
        --host-port) HOST_PORT="$2"; shift ;;
        --container-port) CONTAINER_PORT="$2"; shift ;;
        --checkpoints-dir) CHECKPOINTS_DIR="$2"; shift ;;
        --container-checkpoints-dir) CONTAINER_CHECKPOINTS_DIR="$2"; shift ;;
        --auto-download) AUTO_DOWNLOAD_IF_MISSING="$2"; shift ;;
        --hf-home) HF_HOME_HOST="$2"; shift ;;
        --data-dir) DATA_DIR_HOST="$2"; shift ;;
        --container-data-dir) CONTAINER_DATA_DIR="$2"; shift ;;
        --api) START_API=1 ;;
        --skip-smoke-test) AUTO_SMOKE_TEST=0 ;;
        --smoke-timeout) SMOKE_TIMEOUT_SECONDS="$2"; shift ;;
        --smoke-interval) SMOKE_INTERVAL_SECONDS="$2"; shift ;;
        --no-ui) ENABLE_UI=0 ;;
        --display) DISPLAY_VALUE="$2"; shift ;;
        --xauthority) XAUTHORITY_HOST="$2"; shift ;;
        --detach) DETACH=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown parameter: $1"; usage; exit 1 ;;
    esac
    shift
done

if [ "$REBUILD" -eq 1 ] && [ "$RESTART" -eq 1 ]; then
    echo "ERROR: --rebuild and --restart are mutually exclusive."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BUILD_SCRIPT="$REPO_ROOT/docker/docker_build_describe_anything.sh"
if [ ! -x "$BUILD_SCRIPT" ]; then
    echo "ERROR: Build script missing or not executable: $BUILD_SCRIPT"
    exit 1
fi

API_SCRIPT="$REPO_ROOT/docker/run_describe_anything_api.sh"
if [ "$START_API" -eq 1 ] && [ ! -x "$API_SCRIPT" ]; then
    echo "ERROR: API script missing or not executable: $API_SCRIPT"
    exit 1
fi

SMOKE_SCRIPT="$REPO_ROOT/docker/smoke_test_describe_anything_api.sh"
if [ "$START_API" -eq 1 ] && [ "$AUTO_SMOKE_TEST" -eq 1 ] && [ ! -x "$SMOKE_SCRIPT" ]; then
    echo "ERROR: Smoke-test script missing or not executable: $SMOKE_SCRIPT"
    exit 1
fi

if [ -n "$CHECKPOINTS_DIR" ] && [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "ERROR: checkpoints dir does not exist: $CHECKPOINTS_DIR"
    exit 1
fi

if [ -n "$DATA_DIR_HOST" ]; then
    if [ -d "$DATA_DIR_HOST" ]; then
        echo "Data dir: bind-mount host '$DATA_DIR_HOST' -> container '$CONTAINER_DATA_DIR'"
    else
        echo "WARNING: data dir does not exist: $DATA_DIR_HOST (mount will fail if used)"
    fi
fi

if [ -n "$HF_HOME_HOST" ]; then
    mkdir -p "$HF_HOME_HOST"
    echo "Hugging Face Hub cache: bind-mount host '$HF_HOME_HOST' -> container '$CONTAINER_HF_HOME' (HF_HOME)"
fi

container_exists() {
    podman ps -a -q -f name="^${CONTAINER_NAME}$" | grep -q .
}

container_running() {
    podman ps -q -f name="^${CONTAINER_NAME}$" | grep -q .
}

image_exists() {
    podman image exists "$IMAGE_TAG"
}

append_legacy_nvidia_devices() {
    local -n _target="$1"
    local d
    for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
        if [ -e "$d" ]; then
            _target+=(--device "$d")
        fi
    done
    shopt -s nullglob
    for d in /dev/nvidia[0-9]*; do
        _target+=(--device "$d")
    done
    shopt -u nullglob
}

build_image() {
    local build_args=()
    if [ "$FORCE_FROM_SCRATCH" -eq 1 ]; then
        build_args+=(-f)
    fi
    if [ "$USE_TIMESTAMP_TAG" -eq 1 ]; then
        build_args+=(--timestamp-tag)
    fi
    if [ "$SKIP_LOCAL_BUILD" -eq 1 ]; then
        build_args+=(--no-local)
    fi
    build_args+=(--remote-host "$PRIMARY_REMOTE_HOST")
    build_args+=(--remote-project-dir "$REMOTE_PROJECT_DIR")
    "$BUILD_SCRIPT" "${build_args[@]}"
}

start_new_container() {
    local mount_args=(-v "$REPO_ROOT:/workspace/project:Z")
    if [ -n "$DATA_DIR_HOST" ]; then
        mount_args+=(-v "$DATA_DIR_HOST:$CONTAINER_DATA_DIR:Z")
    fi
    if [ -n "$CHECKPOINTS_DIR" ]; then
        mount_args+=(-v "$CHECKPOINTS_DIR:$CONTAINER_CHECKPOINTS_DIR:Z")
    fi
    if [ -n "$HF_HOME_HOST" ]; then
        mount_args+=(-v "$HF_HOME_HOST:$CONTAINER_HF_HOME:Z")
    fi

    local port_args=(-p "${HOST_PORT}:${CONTAINER_PORT}")

    local env_api_gpu=()
    if [ -n "${DAM_API_CUDA_VISIBLE_DEVICES:-}" ]; then
        env_api_gpu=(-e "DAM_API_CUDA_VISIBLE_DEVICES=${DAM_API_CUDA_VISIBLE_DEVICES}")
    fi

    local env_hf=()
    if [ -n "$HF_HOME_HOST" ]; then
        env_hf+=(-e "HF_HOME=$CONTAINER_HF_HOME")
    fi
    if [ -n "${HF_TOKEN:-}" ]; then
        env_hf+=(-e "HF_TOKEN=${HF_TOKEN}")
    fi

    local env_dam_device=()
    if [ -n "${DAM_DEVICE:-}" ]; then
        env_dam_device+=(-e "DAM_DEVICE=${DAM_DEVICE}")
    fi
    local env_dam_model=()
    if [ -n "$DAM_MODEL_PATH" ]; then
        env_dam_model+=(-e "DAM_MODEL_PATH=${DAM_MODEL_PATH}")
    fi

    local gui_mount_args=()
    local gui_env_args=()
    if [ "$ENABLE_UI" -eq 1 ]; then
        gui_env_args+=(-e "DISPLAY=${DISPLAY_VALUE}")

        if [ -d "$XSOCK_DIR" ]; then
            gui_mount_args+=(-v "$XSOCK_DIR:$XSOCK_DIR:rw")
        else
            echo "WARNING: X socket directory not found on host: $XSOCK_DIR"
        fi

        if [ -n "$XAUTHORITY_HOST" ]; then
            if [ -f "$XAUTHORITY_HOST" ]; then
                gui_mount_args+=(-v "$XAUTHORITY_HOST:/root/.Xauthority:ro")
                gui_env_args+=(-e "XAUTHORITY=/root/.Xauthority")
            else
                echo "WARNING: XAUTHORITY file not found on host: $XAUTHORITY_HOST"
            fi
        fi

        gui_env_args+=(-e "QT_X11_NO_MITSHM=1")
    fi

    local podman_gpu=()
    local gpu_mode="${DAM_PODMAN_NVIDIA_MODE:-auto}"
    case "$gpu_mode" in
        cdi)
            podman_gpu+=(--device nvidia.com/gpu=all)
            echo "GPU passthrough: CDI (--device nvidia.com/gpu=all)"
            ;;
        legacy)
            append_legacy_nvidia_devices podman_gpu
            if [ "${#podman_gpu[@]}" -eq 0 ]; then
                echo "ERROR: DAM_PODMAN_NVIDIA_MODE=legacy but no /dev/nvidia* devices found. Is the NVIDIA driver loaded?" >&2
                exit 1
            fi
            echo "GPU passthrough: legacy (${#podman_gpu[@]} host device(s) under /dev/nvidia*)"
            ;;
        auto|*)
            if [ -e /dev/nvidiactl ]; then
                append_legacy_nvidia_devices podman_gpu
                if [ "${#podman_gpu[@]}" -eq 0 ]; then
                    echo "ERROR: /dev/nvidiactl exists but no devices were added for Podman. Check /dev/nvidia*." >&2
                    exit 1
                fi
                echo "GPU passthrough: legacy /dev/nvidia* (default when host has NVIDIA driver; Podman 3–safe)."
                echo "  For CDI instead: DAM_PODMAN_NVIDIA_MODE=cdi ./docker/run_describe_anything_container.sh ..."
            else
                podman_gpu+=(--device nvidia.com/gpu=all)
                echo "GPU passthrough: CDI (--device nvidia.com/gpu=all; no /dev/nvidiactl on host)."
            fi
            ;;
    esac

    local container_cmd
    if [ "$START_API" -eq 1 ]; then
        container_cmd=(/workspace/project/docker/run_describe_anything_api.sh)
    else
        container_cmd=(bash -lc "trap : TERM INT; sleep infinity & wait")
    fi

    podman run -d \
        --name "$CONTAINER_NAME" \
        "${podman_gpu[@]}" \
        "${port_args[@]}" \
        "${mount_args[@]}" \
        "${gui_mount_args[@]}" \
        -w /workspace/project \
        -e PYTHONUNBUFFERED=1 \
        -e DAM_API_PORT="$CONTAINER_PORT" \
        -e DAM_CHECKPOINTS_DIR="$CONTAINER_CHECKPOINTS_DIR" \
        -e DAM_AUTO_DOWNLOAD_IF_MISSING="$AUTO_DOWNLOAD_IF_MISSING" \
        "${gui_env_args[@]}" \
        "${env_api_gpu[@]}" \
        "${env_hf[@]}" \
        "${env_dam_device[@]}" \
        "${env_dam_model[@]}" \
        "$IMAGE_TAG" \
        "${container_cmd[@]}"
}

if [ "$REBUILD" -eq 1 ]; then
    if container_running; then
        podman stop "$CONTAINER_NAME"
    fi
    if container_exists; then
        podman rm "$CONTAINER_NAME"
    fi
    build_image
else
    if ! image_exists; then
        build_image
    fi
fi

if container_exists; then
    if [ "$RESTART" -eq 1 ]; then
        if container_running; then
            podman stop "$CONTAINER_NAME"
        fi
        podman rm "$CONTAINER_NAME"
        start_new_container
    elif container_running; then
        echo "Container '$CONTAINER_NAME' is already running."
    else
        podman start "$CONTAINER_NAME"
    fi
else
    start_new_container
fi

if ! container_running; then
    echo "ERROR: container '$CONTAINER_NAME' failed to start."
    exit 1
fi

if [ "$START_API" -eq 1 ]; then
    echo "DAM API start requested."
    if [ "$AUTO_SMOKE_TEST" -eq 1 ]; then
        echo "Running API smoke test..."
        "$SMOKE_SCRIPT" \
            --url "http://localhost:${HOST_PORT}" \
            --timeout-seconds "$SMOKE_TIMEOUT_SECONDS" \
            --interval-seconds "$SMOKE_INTERVAL_SECONDS"
    else
        echo "Automatic smoke test skipped (--skip-smoke-test)."
    fi
fi

echo "Host port ${HOST_PORT} -> container ${CONTAINER_PORT} (start the API inside the container to use it)."
if [ "$START_API" -eq 1 ]; then
    echo "DAM API: http://localhost:${HOST_PORT}"
    echo "Manual smoke test: ./docker/smoke_test_describe_anything_api.sh --url http://localhost:${HOST_PORT}"
else
    echo "Idle mode: start the API with:"
    echo "  podman exec -w /workspace/project $CONTAINER_NAME ./docker/run_describe_anything_api.sh"
    echo "Then run: ./docker/smoke_test_describe_anything_api.sh --url http://localhost:${HOST_PORT}"
fi

if [ "$ENABLE_UI" -eq 1 ]; then
    echo "UI support: enabled"
    echo "  DISPLAY=${DISPLAY_VALUE}"
    if [ -d "$XSOCK_DIR" ]; then
        echo "  X socket mount: $XSOCK_DIR"
    fi
    if [ -n "$XAUTHORITY_HOST" ]; then
        echo "  XAUTHORITY host file: $XAUTHORITY_HOST"
    fi
else
    echo "UI support: disabled (--no-ui)"
fi

if [ "$DETACH" -eq 0 ]; then
    podman exec -it -w /workspace/project "$CONTAINER_NAME" bash
fi