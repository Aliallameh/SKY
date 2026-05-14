#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/office/SKY"
PYTHON="$ROOT/.venv_jetson/bin/python"
CONFIG="${SKY_CONFIG:-$ROOT/configs/deploy_jetson_siyi_a8_mini_stage2_engine_1080p.yaml}"
case "$CONFIG" in
    /*) ;;                              # absolute path -> use as-is
    *)  CONFIG="$ROOT/$CONFIG" ;;        # relative path -> resolve under repo root
esac
ENGINE="${SKY_ENGINE:-$ROOT/data/models/yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix/best.engine}"
case "$ENGINE" in
    /*) ;;                               # absolute path -> use as-is
    *)  ENGINE="$ROOT/$ENGINE" ;;        # relative path -> resolve under repo root
esac
OUTPUT_ROOT="$ROOT/data/outputs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUTPUT_ROOT/flight_001_siyi_a8_stage2_tensorrt_1080p_$STAMP"
OPERATOR_LOG="$RUN_DIR/first_flight_siyi_a8_operator.log"

SIYI_A8_CAMERA_IP="${SIYI_A8_CAMERA_IP:-192.168.144.25}"
SIYI_A8_RTSP_URL="${SIYI_A8_RTSP_URL:-rtsp://${SIYI_A8_CAMERA_IP}:8554/main.264}"

OPERATOR_VIEW_MODE="${OPERATOR_VIEW_MODE:-window}"
OPERATOR_VIEW_FULLSCREEN="${OPERATOR_VIEW_FULLSCREEN:-1}"
OPERATOR_VIEW_DISPLAY_WIDTH="${OPERATOR_VIEW_DISPLAY_WIDTH:-1920}"
OPERATOR_VIEW_DISPLAY_HEIGHT="${OPERATOR_VIEW_DISPLAY_HEIGHT:-1080}"
OPERATOR_VIEW_WINDOW_BACKEND="${OPERATOR_VIEW_WINDOW_BACKEND:-gstreamer}"
OPERATOR_VIEW_DISPLAY_FPS="${OPERATOR_VIEW_DISPLAY_FPS:-10}"
OPERATOR_VIEW_PORT="${OPERATOR_VIEW_PORT:-8090}"

# Gimbal follow (SIYI A8 Mini UDP control on 192.168.144.25:37260, CMD_ID 0x07).
# Default OFF. With GIMBAL_FOLLOW_ENABLED=1 and GIMBAL_FOLLOW_DRY_RUN=1 the
# controller only logs proposed yaw/pitch commands. Set GIMBAL_FOLLOW_DRY_RUN=0
# only after bench sign verification with scripts/dev/siyi_gimbal_bench.py.
GIMBAL_FOLLOW_ENABLED="${GIMBAL_FOLLOW_ENABLED:-0}"
GIMBAL_FOLLOW_DRY_RUN="${GIMBAL_FOLLOW_DRY_RUN:-1}"
GIMBAL_FOLLOW_HOST="${GIMBAL_FOLLOW_HOST:-$SIYI_A8_CAMERA_IP}"
GIMBAL_FOLLOW_PORT="${GIMBAL_FOLLOW_PORT:-37260}"
GIMBAL_INVERT_YAW="${GIMBAL_INVERT_YAW:-0}"
GIMBAL_INVERT_PITCH="${GIMBAL_INVERT_PITCH:-0}"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$OPERATOR_LOG") 2>&1

fail() {
    echo
    echo "FAILED: $*"
    echo "Logs so far: $OPERATOR_LOG"
    echo "Press Enter to close this window."
    read -r _ || true
    exit 1
}

case "$OPERATOR_VIEW_MODE" in
    mjpeg|window|both) ;;
    *) fail "Invalid OPERATOR_VIEW_MODE=$OPERATOR_VIEW_MODE. Use mjpeg, window, or both." ;;
esac

echo "SkyScouter SIYI A8 Mini + Insight First Flight Runtime"
echo "Model: Stage 2 multiclass capped review candidate"
echo "Camera: SIYI A8 Mini RTSP"
echo "RTSP URL: $SIYI_A8_RTSP_URL"
echo "Started UTC: $(date -u --iso-8601=seconds)"
echo "Run directory: $RUN_DIR"
echo "Operator view mode: $OPERATOR_VIEW_MODE"
echo "HDMI/window backend: $OPERATOR_VIEW_WINDOW_BACKEND"
echo "HDMI/window output: ${OPERATOR_VIEW_DISPLAY_WIDTH}x${OPERATOR_VIEW_DISPLAY_HEIGHT}, fullscreen=$OPERATOR_VIEW_FULLSCREEN"
if [[ "$GIMBAL_FOLLOW_ENABLED" == "1" || "$GIMBAL_FOLLOW_ENABLED" == "true" ]]; then
    if [[ "$GIMBAL_FOLLOW_DRY_RUN" == "1" || "$GIMBAL_FOLLOW_DRY_RUN" == "true" ]]; then
        echo "Gimbal follow: ENABLED (DRY_RUN — no UDP sent) target=$GIMBAL_FOLLOW_HOST:$GIMBAL_FOLLOW_PORT"
    else
        echo "Gimbal follow: ENABLED (LIVE UDP) target=$GIMBAL_FOLLOW_HOST:$GIMBAL_FOLLOW_PORT  invert_yaw=$GIMBAL_INVERT_YAW invert_pitch=$GIMBAL_INVERT_PITCH"
    fi
else
    echo "Gimbal follow: DISABLED"
fi
echo

cd "$ROOT"

echo "Checking required files..."
[[ -x "$PYTHON" ]] || fail "Jetson venv Python not found or not executable: $PYTHON"
[[ -f "$CONFIG" ]] || fail "Runtime config missing: $CONFIG"
[[ -f "$ENGINE" ]] || fail "TensorRT engine missing: $ENGINE"

echo "Checking A8 Mini network reachability..."
if ping -c 1 -W 2 "$SIYI_A8_CAMERA_IP" >/dev/null 2>&1; then
    echo "A8 Mini ping: PASS ($SIYI_A8_CAMERA_IP)"
else
    echo "A8 Mini ping: FAIL ($SIYI_A8_CAMERA_IP)"
    echo
    echo "The Jetson Ethernet interface must be configured on the A8 subnet."
    echo "Known A8 Mini default camera IP: $SIYI_A8_CAMERA_IP"
    echo "Recommended Jetson Ethernet static IP: 192.168.144.10/24"
    echo
    echo "Example NetworkManager command:"
    echo "  sudo nmcli con mod 'Wired connection 1' ipv4.method manual ipv4.addresses 192.168.144.10/24 ipv4.gateway '' ipv4.dns ''"
    echo "  sudo nmcli con up 'Wired connection 1'"
    fail "A8 Mini is not reachable. Fix Ethernet IP/wiring/power before running RTSP inference."
fi
echo

echo "Setting Jetson flight power mode..."
echo "Using nvpmodel mode 0: 15W. This avoids MAXN_SUPER over-current risk during flight."
if sudo -n nvpmodel -m 0; then
    sudo -n nvpmodel -q || true
else
    echo "WARNING: could not set 15W power mode. Continuing with current nvpmodel mode."
    sudo -n nvpmodel -q || true
fi
echo "Not enabling jetson_clocks for flight. Stability and power margin are more important than peak benchmark speed."
echo

echo "Running preflight with RTSP probe..."
"$PYTHON" scripts/dev/jetson_preflight_check.py \
    --config "$CONFIG" \
    --engine "$ENGINE" \
    --probe-rtsp \
    --rtsp-url "$SIYI_A8_RTSP_URL" \
    --output-dir "$RUN_DIR/preflight" || fail "Preflight failed"

echo
echo "Preflight passed."
echo "Verify the Insight receiver sees the Jetson display before takeoff."
echo "After landing, press Ctrl+C once. The pipeline will close video writers and finalize manifest.json."
echo

PIPELINE_CMD=(
    "$PYTHON" scripts/run_pipeline.py
    --config "$CONFIG"
    --source-url "$SIYI_A8_RTSP_URL"
    --output "$RUN_DIR"
    --operator-view
    --operator-view-mode "$OPERATOR_VIEW_MODE"
    --operator-view-host 0.0.0.0
    --operator-view-port "$OPERATOR_VIEW_PORT"
    --operator-view-max-width 1280
    --operator-view-jpeg-quality 70
    --operator-view-display-width "$OPERATOR_VIEW_DISPLAY_WIDTH"
    --operator-view-display-height "$OPERATOR_VIEW_DISPLAY_HEIGHT"
    --operator-view-window-backend "$OPERATOR_VIEW_WINDOW_BACKEND"
    --operator-view-display-fps "$OPERATOR_VIEW_DISPLAY_FPS"
)

if [[ "$OPERATOR_VIEW_FULLSCREEN" == "1" || "$OPERATOR_VIEW_FULLSCREEN" == "true" ]]; then
    PIPELINE_CMD+=(--operator-view-fullscreen)
fi

if [[ "$GIMBAL_FOLLOW_ENABLED" == "1" || "$GIMBAL_FOLLOW_ENABLED" == "true" ]]; then
    PIPELINE_CMD+=(--gimbal-follow-enabled)
    PIPELINE_CMD+=(--gimbal-host "$GIMBAL_FOLLOW_HOST")
    PIPELINE_CMD+=(--gimbal-port "$GIMBAL_FOLLOW_PORT")
    if [[ "$GIMBAL_FOLLOW_DRY_RUN" == "1" || "$GIMBAL_FOLLOW_DRY_RUN" == "true" ]]; then
        PIPELINE_CMD+=(--gimbal-follow-dry-run)
    else
        PIPELINE_CMD+=(--gimbal-follow-live)
    fi
    if [[ "$GIMBAL_INVERT_YAW" == "1" || "$GIMBAL_INVERT_YAW" == "true" ]]; then
        PIPELINE_CMD+=(--gimbal-invert-yaw)
    fi
    if [[ "$GIMBAL_INVERT_PITCH" == "1" || "$GIMBAL_INVERT_PITCH" == "true" ]]; then
        PIPELINE_CMD+=(--gimbal-invert-pitch)
    fi
fi

set +e
"${PIPELINE_CMD[@]}"
PIPELINE_STATUS=$?
set -e

echo
echo "Pipeline exited with status: $PIPELINE_STATUS"
echo "Finished UTC: $(date -u --iso-8601=seconds)"
echo

echo "Output files:"
find "$RUN_DIR" -maxdepth 2 -type f -printf "%p %s bytes\n" | sort
echo

if [[ -f "$RUN_DIR/manifest.json" ]]; then
    echo "Manifest status:"
    "$PYTHON" - <<'PY' "$RUN_DIR/manifest.json"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ["run_id", "status", "frame_count", "detections_total", "tracks_created", "started_utc", "ended_utc"]:
    print(f"{key}: {manifest.get(key)}")
PY
else
    echo "WARNING: manifest.json was not found."
fi

echo
echo "SIYI A8 Mini run directory:"
echo "$RUN_DIR"
echo
echo "Press Enter to close this window."
read -r _ || true
exit "$PIPELINE_STATUS"
