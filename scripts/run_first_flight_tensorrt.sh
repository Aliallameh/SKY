#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/office/SKY"
PYTHON="$ROOT/.venv_jetson/bin/python"
CONFIG="$ROOT/configs/deploy_jetson_yolo11s_stage2_multiclass_capped_engine_1080p.yaml"
ENGINE="$ROOT/data/models/yolo11s_airborne_stage2_multiclass_capped_aod4conf_b16w4_nomix/best.engine"
OUTPUT_ROOT="$ROOT/data/outputs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUTPUT_ROOT/flight_001_stage2_tensorrt_1080p_$STAMP"
OPERATOR_LOG="$RUN_DIR/first_flight_operator.log"
OPERATOR_VIEW_PORT="${OPERATOR_VIEW_PORT:-8090}"

mkdir -p "$RUN_DIR"

exec > >(tee -a "$OPERATOR_LOG") 2>&1

echo "SkyScouter First Flight TensorRT Runtime"
echo "Model: Stage 2 multiclass capped review candidate"
echo "Capture mode: 1920x1080 MJPG 30 fps"
echo "Started UTC: $(date -u --iso-8601=seconds)"
echo "Run directory: $RUN_DIR"
echo "Live operator view: http://<jetson-ip>:$OPERATOR_VIEW_PORT/"
echo "Local operator view: http://127.0.0.1:$OPERATOR_VIEW_PORT/"
echo

cd "$ROOT"

fail() {
    echo
    echo "FAILED: $*"
    echo "Logs so far: $OPERATOR_LOG"
    echo "Press Enter to close this window."
    read -r _ || true
    exit 1
}

echo "Checking required files..."
[[ -x "$PYTHON" ]] || fail "Jetson venv Python not found or not executable: $PYTHON"
[[ -f "$CONFIG" ]] || fail "Runtime config missing: $CONFIG"
[[ -f "$ENGINE" ]] || fail "TensorRT engine missing: $ENGINE"

echo "Engine:"
ls -lh "$ENGINE"
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

echo "Running preflight with camera probe..."
"$PYTHON" scripts/dev/jetson_preflight_check.py \
    --camera-device /dev/video0 \
    --camera-index 0 \
    --config "$CONFIG" \
    --engine "$ENGINE" \
    --probe-camera \
    --output-dir "$RUN_DIR/preflight" || fail "Preflight failed"

echo
echo "Preflight passed."
echo
echo "RECORDING STARTING NOW"
echo "Keep this terminal open during flight."
echo "Open the live operator view in a browser before takeoff if needed."
echo "After landing, press Ctrl+C once. The pipeline will close video writers and finalize manifest.json."
echo

set +e
"$PYTHON" scripts/run_pipeline.py \
    --config "$CONFIG" \
    --output "$RUN_DIR" \
    --operator-view \
    --operator-view-mode mjpeg \
    --operator-view-host 0.0.0.0 \
    --operator-view-port "$OPERATOR_VIEW_PORT" \
    --operator-view-max-width 1280 \
    --operator-view-jpeg-quality 70
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
echo "First flight run directory:"
echo "$RUN_DIR"
echo
echo "Press Enter to close this window."
read -r _ || true
exit "$PIPELINE_STATUS"
