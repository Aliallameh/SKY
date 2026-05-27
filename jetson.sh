#!/bin/bash
# =============================================================================
# SkyScouter — Jetson Setup & Launcher
# =============================================================================
# Single entry point for a fresh Jetson Orin.  Run once to set up the
# environment, then use the menu to run any pipeline or tool.
#
# Field-friendly: once the venv exists, `./jetson.sh` goes STRAIGHT to the
# menu without touching the network.  This means the live pipeline can be
# launched without internet (e.g. in the field).  Dependency sync is a
# manual menu action (option 11) for when you have connectivity.
#
# Usage:
#   chmod +x jetson.sh
#   ./jetson.sh          # if venv exists: menu only (offline-safe)
#                        # if venv missing: full setup, then menu
#   ./jetson.sh setup    # force full environment setup (needs internet)
#   ./jetson.sh verify   # verify environment only
#   ./jetson.sh run <N>  # non-interactive: run menu option N
#                        # Model via env: SKYSCOUTER_MODEL=<dirname>
#                        # Field workflow: SSH in from laptop, run this.
#   ./jetson.sh autostart  # systemd service install/enable/disable/logs
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv_jetson"
PY="$VENV/bin/python3"
# libcudss.so.0 is required by Jetson AI Lab torch but not shipped by JetPack.
# Only this directory is prepended; all other CUDA libs use system JetPack paths
# to avoid cuBLAS/cuDNN version conflicts.
CUDSS_LIB="$VENV/lib/python3.10/site-packages/nvidia/cu12/lib"
export LD_LIBRARY_PATH="$CUDSS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Colours ──────────────────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'
B='\033[0;34m'; C='\033[0;36m'; W='\033[1;37m'; N='\033[0m'

ok()   { echo -e "${G}  ✓ $*${N}"; }
info() { echo -e "${B}  → $*${N}"; }
warn() { echo -e "${Y}  ⚠ $*${N}"; }
fail() { echo -e "${R}  ✗ $*${N}"; exit 1; }
hdr()  { echo -e "\n${W}━━━  $*  ━━━${N}"; }

# ── Helper: run python inside the venv with correct LD_LIBRARY_PATH ──────────
pyrun() { LD_LIBRARY_PATH="$CUDSS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PY" "$@"; }
piprun() { pyrun -m pip "$@"; }

# =============================================================================
# 1. ENVIRONMENT CHECKS
# =============================================================================
check_jetson() {
    hdr "Jetson platform check"
    if [ ! -f /etc/nv_tegra_release ]; then
        warn "Not a Jetson device — continuing anyway."
    else
        L4T=$(grep "REVISION:" /etc/nv_tegra_release | grep -oP 'REVISION: \K[0-9.]+')
        ok "L4T R36.$L4T detected (JetPack 6.x)"
    fi

    CUDA_VER=$(ls /usr/local/ | grep "^cuda-" | sort -V | tail -1 | sed 's/cuda-//')
    if [ -z "$CUDA_VER" ]; then
        warn "Could not detect CUDA version from /usr/local/cuda-*"
    else
        ok "System CUDA $CUDA_VER found"
    fi

    # gstreamer1.0-plugins-bad provides h264parse / h265parse which are
    # required for the nvv4l2decoder hardware RTSP decode path.
    # Safe to install — standard Ubuntu package, no JetPack CUDA conflict.
    if ! gst-inspect-1.0 h264parse &>/dev/null 2>&1; then
        info "gstreamer1.0-plugins-bad not found — installing (requires sudo)..."
        sudo apt-get install -y gstreamer1.0-plugins-bad
        ok "gstreamer1.0-plugins-bad installed"
    else
        ok "gstreamer1.0-plugins-bad present (h264parse / h265parse available)"
    fi
}

# =============================================================================
# 2. VENV HEALTH CHECK
# =============================================================================
venv_is_healthy() {
    # Venv must exist, python must be runnable, and the skyscouter editable
    # install must point to THIS repo (not a leftover from another machine).
    [ -x "$PY" ] || return 1
    "$PY" -c "import sys; sys.exit(0)" 2>/dev/null || return 1
    # pip >= 22 shows "Editable project location:"; older pip shows the source
    # dir under "Location:".  Try the explicit editable field first.
    EDITABLE_LOC=$(piprun show skyscouter 2>/dev/null \
        | awk '/^Editable project location:/{print $NF; exit}')
    if [ -z "$EDITABLE_LOC" ]; then
        EDITABLE_LOC=$(piprun show skyscouter 2>/dev/null \
            | awk '/^Location:/{print $NF; exit}')
    fi
    [ "$EDITABLE_LOC" = "$REPO" ] || return 1
    return 0
}

# =============================================================================
# 3. SETUP
# =============================================================================
setup_venv() {
    hdr "Virtual environment"
    if venv_is_healthy; then
        ok "Venv at $VENV is healthy — skipping recreate"
        return 0
    fi

    if [ -d "$VENV" ]; then
        warn "Removing stale venv (was created on a different machine/path)..."
        rm -rf "$VENV"
    fi

    # python3.10-venv is not always present on a fresh Ubuntu/JetPack image.
    # Test ensurepip directly — dpkg -l can show the package as "known" even
    # when it is not properly installed, so we test the real capability.
    if ! python3 -m ensurepip --version &>/dev/null 2>&1; then
        warn "python3.10-venv / ensurepip not available — installing (requires sudo)..."
        sudo apt-get install -y python3.10-venv
        ok "python3.10-venv installed"
    fi

    info "Creating venv with --system-site-packages (inherits JetPack libs)..."
    python3 -m venv "$VENV" --system-site-packages
    ok "Venv created at $VENV"

    info "Upgrading pip / setuptools / wheel..."
    piprun install --upgrade pip setuptools wheel
    ok "pip upgraded"
}

install_torch() {
    hdr "PyTorch (Jetson AI Lab — jp6/cu126)"

    if pyrun -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        VER=$(pyrun -c "import torch; print(torch.__version__)" 2>/dev/null)
        ok "torch $VER already installed with CUDA — skipping"
        return 0
    fi

    info "Installing torch + torchvision from https://pypi.jetson-ai-lab.io/jp6/cu126 ..."
    info "(This is the ONLY correct source for Jetson Orin sm_87 kernels)"
    piprun install torch torchvision \
        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
    ok "torch installed"
}

install_cudss() {
    hdr "CUDA cuDSS (libcudss.so.0)"
    # The Jetson AI Lab torch links against libcudss which JetPack does NOT ship.
    # Install ONLY the cudss pip package — do NOT install nvidia-cublas-cu12,
    # nvidia-cuda-runtime-cu12, etc. Those would shadow JetPack's system CUDA
    # and cause CUBLAS_STATUS_ALLOC_FAILED.
    if [ -f "$CUDSS_LIB/libcudss.so.0" ]; then
        ok "libcudss.so.0 already present — skipping"
        return 0
    fi
    info "Installing nvidia-cudss-cu12 (provides libcudss.so.0)..."
    # --no-deps is CRITICAL: nvidia-cudss-cu12 declares cuda-toolkit as a
    # dependency which pulls nvidia-cublas-cu12, nvidia-cuda-nvrtc-cu12, etc.
    # Those packages shadow JetPack's system CUDA libs and cause
    # CUBLAS_STATUS_ALLOC_FAILED at runtime.  We only need the .so file itself;
    # all other CUDA libs are already provided by the JetPack installation.
    piprun install --no-deps nvidia-cudss-cu12
    ok "nvidia-cudss-cu12 installed"
}

install_requirements() {
    hdr "Project requirements"

    # protobuf: system version is too old (missing google.protobuf.internal.builder)
    # needed by onnx during TRT engine export
    info "Installing protobuf>=3.20 (system version too old for onnx)..."
    piprun install "protobuf>=3.20"

    # onnx + onnxslim: required for TRT engine export (PyTorch → ONNX → TRT).
    # Install explicitly BEFORE ultralytics so ultralytics does not attempt its
    # broken auto-update which also tries onnxruntime-gpu (no aarch64 PyPI wheel).
    info "Installing onnx + onnxslim (needed for TRT engine export)..."
    piprun install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.71"

    # Explicit uninstall of any stale pip CUDA libs that may have crept in from
    # a failed install (nvidia-cublas-cu12 etc. shadow JetPack libs and break
    # cuBLAS). Only nvidia-cudss-cu12 is allowed.
    CONFLICT_PKGS=(
        nvidia-cublas-cu12
        nvidia-cuda-cupti-cu12
        nvidia-cuda-nvrtc-cu12
        nvidia-cuda-runtime-cu12
        nvidia-cufile-cu12
        nvidia-curand-cu12
        nvidia-cusparselt-cu12
        nvidia-nccl-cu12
        nvidia-nvjitlink-cu12
        nvidia-nvshmem-cu12
        nvidia-nvtx-cu12
        cuda-toolkit
        cuda-pathfinder
    )
    INSTALLED_CONFLICTS=()
    for pkg in "${CONFLICT_PKGS[@]}"; do
        piprun show "$pkg" &>/dev/null && INSTALLED_CONFLICTS+=("$pkg")
    done
    if [ ${#INSTALLED_CONFLICTS[@]} -gt 0 ]; then
        warn "Removing conflicting pip CUDA packages: ${INSTALLED_CONFLICTS[*]}"
        piprun uninstall -y "${INSTALLED_CONFLICTS[@]}"
    fi

    # requirements-jetson.txt contains all other deps and '-e .' for the
    # editable skyscouter package install.
    info "Installing requirements-jetson.txt..."
    piprun install -r "$REPO/requirements-jetson.txt"

    ok "All requirements installed"
}

verify_env() {
    hdr "Environment verification"
    pyrun -c "
import sys, torch, cv2, ultralytics, tensorrt
cap = torch.cuda.get_device_capability()
sm  = f'sm_{cap[0]}{cap[1]}'
assert torch.cuda.is_available(),      'CUDA not available in torch'
assert cap == (8, 7),                  f'Expected sm_87 (Orin), got {sm}'
x = torch.randn(3,3).cuda()
torch.mm(x, x)  # confirm cuBLAS works
print(f'  torch        {torch.__version__}  CUDA:{torch.version.cuda}  GPU:{sm}')
print(f'  cv2          {cv2.__version__}')
print(f'  ultralytics  {ultralytics.__version__}')
print(f'  tensorrt     {tensorrt.__version__}')
print(f'  python       {sys.version.split()[0]}')
"
    ok "All imports verified"
}

check_engine() {
    hdr "TRT engine check"
    # Use ^\s*weights: to skip comment lines (e.g. "# base_weights: yolov26n.pt")
    # which also contain "weights:" and would be matched by a plain grep.
    CFG_ENGINE=$(grep -E "^\s*weights:" "$REPO/configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml" \
                 | head -1 | awk '{print $2}' | tr -d '"')
    ENGINE_PATH="$REPO/$CFG_ENGINE"

    if [ -f "$ENGINE_PATH" ]; then
        MANIFEST="${ENGINE_PATH%.engine}.export_manifest.json"
        if [ -f "$MANIFEST" ]; then
            BUILT_ON=$("$PY" -c "
import json
m = json.load(open('$MANIFEST'))
print(m['platform'].get('jetson_l4t_release','?').split('REVISION:')[1].split(',')[0].strip()
      if 'REVISION:' in m['platform'].get('jetson_l4t_release','') else '?')
" 2>/dev/null)
            THIS_L4T=$(grep "REVISION:" /etc/nv_tegra_release | grep -oP 'REVISION: \K[0-9.]+')
            if [ "$BUILT_ON" = "$THIS_L4T" ]; then
                ok "Engine $ENGINE_PATH — built on this Jetson (R36.$THIS_L4T) ✓"
            else
                warn "Engine was built on R36.$BUILT_ON, this Jetson is R36.$THIS_L4T"
                warn "Detections may be wrong. Run option 4 to rebuild the engine."
            fi
        else
            ok "Engine $ENGINE_PATH exists"
        fi
    else
        warn "Engine not found: $ENGINE_PATH"
        warn "Run option 4 from the menu to build it (~6 min)."
    fi
}

do_setup() {
    echo -e "\n${C}╔══════════════════════════════════════╗"
    echo -e "║   SkyScouter Jetson Environment      ║"
    echo -e "╚══════════════════════════════════════╝${N}"
    check_jetson
    setup_venv
    install_torch
    install_cudss
    install_requirements
    verify_env
    check_engine
    echo -e "\n${G}Setup complete. Run ./jetson.sh to launch the menu.${N}\n"
}

# =============================================================================
# 4. NETWORK HELPERS
# =============================================================================
configure_network() {
    hdr "Network — SIYI A8 Mini Ethernet"
    echo "The SIYI A8 Mini has fixed IP 192.168.144.25."
    echo "The Jetson needs a DIFFERENT IP in the same subnet (e.g. 192.168.144.10)."
    echo ""
    nmcli connection show 2>/dev/null | grep -i ethernet || true
    echo ""
    read -rp "Enter Jetson Ethernet connection name (from above): " CONN
    read -rp "Enter Jetson IP to use [192.168.144.10]: " JIP
    JIP="${JIP:-192.168.144.10}"
    sudo nmcli connection modify "$CONN" ipv4.addresses "$JIP/24"
    sudo nmcli connection modify "$CONN" ipv4.gateway "192.168.144.1"
    sudo nmcli connection modify "$CONN" ipv4.method manual
    sudo nmcli connection up "$CONN"
    echo ""
    ok "Interface reconfigured. Testing camera reachability..."
    sleep 1
    if ping -c 2 -W 2 192.168.144.25 &>/dev/null; then
        CAMERA_MAC=$(arp -n 192.168.144.25 | awk '/ether/{print $3}')
        ok "Camera reachable — MAC $CAMERA_MAC"
    else
        fail "Camera still not reachable. Check cable and camera power."
    fi
}

# =============================================================================
# 5. MODEL SCANNER & SELECTOR
# =============================================================================
# Globals set by select_model():
SELECTED_MODEL_DIR=""
SELECTED_PT_FILE=""
# Global set by make_temp_config():
TEMP_CONFIG_PATH=""

select_model() {
    # Scans data/models/, prints a status table, and sets SELECTED_MODEL_DIR.
    # PURPOSE: "pipeline" = need .engine; "export" = need real .pt
    #
    # Non-interactive bypass: if SKYSCOUTER_MODEL is set to a directory name
    # under data/models/, skip the picker entirely.  Used by `./jetson.sh run`
    # and by the systemd autostart service.
    local PURPOSE="${1:-pipeline}"
    local MODELS_DIR="$REPO/data/models"

    [ -d "$MODELS_DIR" ] || fail "Models directory not found: $MODELS_DIR"

    if [ -n "${SKYSCOUTER_MODEL:-}" ]; then
        local override_dir="$MODELS_DIR/$SKYSCOUTER_MODEL"
        if [ -d "$override_dir" ]; then
            SELECTED_MODEL_DIR="$override_dir"
            SELECTED_PT_FILE=""
            for f in "$override_dir/best.pt" "$override_dir/last.pt"; do
                [ -f "$f" ] && SELECTED_PT_FILE="$f" && break
            done
            ok "Model from \$SKYSCOUTER_MODEL: $SKYSCOUTER_MODEL"
            return 0
        else
            warn "\$SKYSCOUTER_MODEL=$SKYSCOUTER_MODEL not found under $MODELS_DIR"
            warn "Falling back to interactive picker."
        fi
    fi

    local THIS_L4T
    THIS_L4T=$(grep -oP 'REVISION: \K[0-9.]+' /etc/nv_tegra_release 2>/dev/null || echo "?")

    local -a ALL_DIRS=()
    while IFS= read -r d; do ALL_DIRS+=("$d"); done \
        < <(find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

    [ ${#ALL_DIRS[@]} -gt 0 ] || fail "No model directories found under $MODELS_DIR"

    hdr "Available models — $MODELS_DIR"
    echo ""
    printf "  ${W}%-3s  %-40s  %-6s  %-7s  %s${N}\n" \
        "#" "Model directory" ".pt" ".engine" "Engine / L4T status"
    echo "  ---  ----------------------------------------  ------  -------  --------------------------"

    local -a SHOW_DIRS=()
    local -a SHOW_PT=()
    local n=1

    for dir in "${ALL_DIRS[@]}"; do
        local name; name=$(basename "$dir")

        # ── .pt status ───────────────────────────────────────────────────────
        local pt_label="  ✗"
        local pt_path=""
        for f in "$dir/best.pt" "$dir/last.pt"; do
            if [ -f "$f" ]; then
                local sz; sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
                if [ "$sz" -gt 10000 ]; then
                    pt_label="  ✓"
                    [ "$(basename "$f")" = "last.pt" ] && pt_label=" last"
                    pt_path="$f"; break
                else
                    pt_label=" LFS"   # git-LFS pointer, not real weights
                fi
            fi
        done

        # ── .engine status ───────────────────────────────────────────────────
        local eng_label="  ✗"
        local eng_info="-"
        if [ -f "$dir/best.engine" ]; then
            eng_label="  ✓"
            local manifest="$dir/best.export_manifest.json"
            if [ -f "$manifest" ] && [ -x "$PY" ]; then
                local built_on
                built_on=$("$PY" -c "
import json, re
try:
    m = json.load(open('$manifest'))
    r = m['platform'].get('jetson_l4t_release', '')
    v = re.search(r'REVISION: ([0-9.]+)', r)
    print(v.group(1) if v else '?')
except Exception: print('?')
" 2>/dev/null || echo "?")
                if [ "$built_on" = "$THIS_L4T" ]; then
                    eng_info="R36.$built_on ✓"
                else
                    eng_info="R36.$built_on ≠ R36.$THIS_L4T ⚠ rebuild!"
                fi
            else
                eng_info="exists (no manifest)"
            fi
        fi

        # ── filter by purpose ────────────────────────────────────────────────
        if [ "$PURPOSE" = "export" ] && [ -z "$pt_path" ]; then
            continue   # needs a real .pt — skip dirs with only pointer/no .pt
        fi
        [ "$eng_label" = "  ✗" ] && [ "$PURPOSE" = "pipeline" ] && \
            eng_info="${Y}no engine — export first${N}"

        printf "  ${C}%-3s${N}  %-40s  %-6s  %-7s  " "$n)" "$name" "$pt_label" "$eng_label"
        echo -e "$eng_info"

        SHOW_DIRS+=("$dir")
        SHOW_PT+=("$pt_path")
        n=$((n+1))
    done
    echo ""

    if [ ${#SHOW_DIRS[@]} -eq 0 ]; then
        if [ "$PURPOSE" = "export" ]; then
            fail "No models with real .pt weights found."$'\n'"       Run:  git lfs pull   to download LFS-tracked weights."
        else
            fail "No model directories found under $MODELS_DIR"
        fi
    fi

    local SEL
    while true; do
        read -rp "  Select model [1-$((n-1))]: " SEL
        [[ "$SEL" =~ ^[0-9]+$ ]] && [ "$SEL" -ge 1 ] && [ "$SEL" -le $((n-1)) ] && break
        warn "Enter a number between 1 and $((n-1))"
    done

    SELECTED_MODEL_DIR="${SHOW_DIRS[$((SEL-1))]}"
    SELECTED_PT_FILE="${SHOW_PT[$((SEL-1))]}"
    ok "Selected: $(basename "$SELECTED_MODEL_DIR")"
}

make_temp_config() {
    # Builds a temp YAML (base deploy config with engine path + imgsz patched).
    # Sets TEMP_CONFIG_PATH.  Caller must rm it after the pipeline exits.
    local MODEL_DIR="$1"
    local ENGINE="$MODEL_DIR/best.engine"
    local BASE_CFG="$REPO/configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml"

    if [ ! -f "$ENGINE" ]; then
        warn "No engine in $(basename "$MODEL_DIR") — need to export first."
        read -rp "  Export TRT engine now? [Y/n]: " ANS
        ANS="${ANS:-Y}"
        if [[ "$ANS" =~ ^[Yy] ]]; then
            _do_export_engine "$MODEL_DIR"
        else
            fail "Cannot run pipeline without a TRT engine."
        fi
    fi

    # Read imgsz from export manifest if present
    local IMGSZ=1024
    local MANIFEST="$MODEL_DIR/best.export_manifest.json"
    if [ -f "$MANIFEST" ] && [ -x "$PY" ]; then
        local detected
        detected=$("$PY" -c "
import json
try:
    m = json.load(open('$MANIFEST'))
    v = m.get('export_args', {}).get('imgsz') or m.get('imgsz')
    print(int(v) if v else 1024)
except Exception: print(1024)
" 2>/dev/null || echo "1024")
        [ -n "$detected" ] && IMGSZ="$detected"
    fi

    local RELPATH="${ENGINE#$REPO/}"
    TEMP_CONFIG_PATH=$(mktemp /tmp/skyscouter_XXXXXX.yaml)

    # Patch only the detector weights and input_size lines (anchored with \s*)
    sed -E \
        -e "s|^(\s*weights:).*|\1 \"$RELPATH\"|" \
        -e "s|^(\s*input_size:).*|\1 $IMGSZ|" \
        "$BASE_CFG" > "$TEMP_CONFIG_PATH"

    info "Model  → $(basename "$MODEL_DIR")"
    info "Engine → $RELPATH  (imgsz=$IMGSZ)"
}

# =============================================================================
# 6. ENGINE EXPORT
# =============================================================================
_do_export_engine() {
    # Internal: run the actual TRT export for a given model directory.
    local MODEL_DIR="$1"
    local PT="$SELECTED_PT_FILE"
    # If called internally (not from export_engine menu), find best.pt ourselves
    if [ -z "$PT" ] || [ ! -f "$PT" ]; then
        PT=$(find "$MODEL_DIR" \( -name "best.pt" -o -name "last.pt" \) | sort | head -1)
    fi
    [ -f "$PT" ] || fail "No .pt file found in $MODEL_DIR"
    local sz; sz=$(stat -c%s "$PT")
    [ "$sz" -gt 10000 ] || \
        fail "$(basename "$PT") is a git-LFS pointer ($sz bytes)."$'\n'"       Run:  git lfs pull"

    read -rp "  Input image size [1024]: " IMGSZ; IMGSZ="${IMGSZ:-1024}"
    read -rp "  Precision — FP16 (faster) or FP32? [fp16]: " PREC; PREC="${PREC:-fp16}"
    local HALF_FLAG=""
    [[ "${PREC,,}" =~ ^(fp16|half)$ ]] && HALF_FLAG="--half"
    local PREC_LABEL="FP32"; [ -n "$HALF_FLAG" ] && PREC_LABEL="FP16"
    info "Exporting $PREC_LABEL engine from $(basename "$PT") at imgsz=$IMGSZ (~4-6 min)..."
    pyrun "$REPO/scripts/export_tensorrt.py" \
        --weights "$PT" \
        --imgsz "$IMGSZ" \
        $HALF_FLAG
    ok "Engine → $MODEL_DIR/best.engine"
}

export_engine() {
    hdr "Export TensorRT engine"
    select_model "export"
    _do_export_engine "$SELECTED_MODEL_DIR"
    echo ""
    read -rp "  Set as default config engine? [Y/n]: " ANS
    ANS="${ANS:-Y}"
    if [[ "$ANS" =~ ^[Yy] ]]; then
        local RELPATH="${SELECTED_MODEL_DIR#$REPO/}/best.engine"
        local IMGSZ
        IMGSZ=$(grep -oP '"imgsz":\s*\K[0-9]+' \
            "$SELECTED_MODEL_DIR/best.export_manifest.json" 2>/dev/null || echo "1024")
        local BASE_CFG="$REPO/configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml"
        sed -i -E "s|^(\s*weights:).*|\1 \"$RELPATH\"|"  "$BASE_CFG"
        sed -i -E "s|^(\s*input_size:).*|\1 $IMGSZ|"     "$BASE_CFG"
        ok "Default config updated → $RELPATH  (imgsz=$IMGSZ)"
    fi
}

# =============================================================================
# 7. PIPELINE LAUNCHERS
# =============================================================================
default_output() {
    echo "$REPO/data/outputs/${1}_$(date -u +%Y%m%dT%H%M%SZ)"
}

run_pipeline() {
    local EXTRA_ARGS=("$@")
    select_model "pipeline"
    make_temp_config "$SELECTED_MODEL_DIR"
    local CFG="$TEMP_CONFIG_PATH"
    local OUTDIR
    OUTDIR=$(default_output "live_$(basename "$SELECTED_MODEL_DIR")")
    info "Output → $OUTDIR"
    cd "$REPO"
    pyrun scripts/run_pipeline.py \
        --config "$CFG" \
        --output "$OUTDIR" \
        "${EXTRA_ARGS[@]}" || true
    rm -f "$CFG"
}

run_preflight() {
    hdr "Preflight check"
    cd "$REPO"
    pyrun scripts/dev/jetson_preflight_check.py \
        --config configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml \
        --probe-rtsp
}

run_smoke() {
    hdr "Smoke test (≈30-second run — press Ctrl+C to stop)"
    info "Gimbal and display disabled for smoke. Ctrl+C to stop."
    select_model "pipeline"
    make_temp_config "$SELECTED_MODEL_DIR"
    local CFG="$TEMP_CONFIG_PATH"
    local OUTDIR
    OUTDIR=$(default_output "smoke_$(basename "$SELECTED_MODEL_DIR")")
    info "Output → $OUTDIR"
    cd "$REPO"
    pyrun scripts/run_pipeline.py \
        --config "$CFG" \
        --output "$OUTDIR" \
        --no-operator-view \
        --no-gimbal-follow || true
    rm -f "$CFG"
}

get_jetson_ip() {
    # Find the IP in the camera subnet on any interface (not hardcoded to eno1)
    ip addr show 2>/dev/null | awk '/inet 192\.168\.144\./{print $2}' | cut -d/ -f1 | head -1
}

# =============================================================================
# 7. MENU
# =============================================================================
show_menu() {
    while true; do
        JETSON_IP=$(get_jetson_ip)
        MODEL_COUNT=$(find "$REPO/data/models" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        ENGINE_COUNT=$(find "$REPO/data/models" -name "best.engine" 2>/dev/null | wc -l)

        echo -e "\n${C}╔══════════════════════════════════════════════════════╗"
        echo -e "║            SkyScouter — Jetson Launcher              ║"
        echo -e "╠══════════════════════════════════════════════════════╣"
        printf  "║  Models : %-43s║\n" "$MODEL_COUNT dirs  ($ENGINE_COUNT with .engine)  — selected at runtime"
        printf  "║  Jetson : %-43s║\n" "${JETSON_IP:-not configured}  →  Camera: 192.168.144.25"
        echo -e "╠══════════════════════════════════════════════════════╣"
        echo -e "║  PIPELINE                                            ║"
        echo -e "║    1) Run live pipeline  (TRT — no display)          ║"
        echo -e "║    2) Run live pipeline  (TRT — MJPEG stream)        ║"
        echo -e "║    3) Run live pipeline  (TRT — OpenCV window)       ║"
        echo -e "║    4) Run live pipeline  (gimbal follow DISABLED)    ║"
        echo -e "║    5) Run pipeline + FC dry-run    (no commands sent)║"
        echo -e "║    6) Run pipeline + FC LIVE       ⚠ REAL FLIGHT     ║"
        echo -e "╠══════════════════════════════════════════════════════╣"
        echo -e "║  TOOLS                                               ║"
        echo -e "║    7) Export TRT engine from .pt weights             ║"
        echo -e "║    8) Preflight check  (camera + deps + config)      ║"
        echo -e "║    9) Smoke test  (30-second run)                    ║"
        echo -e "║   10) Verify environment                             ║"
        echo -e "║   11) Configure Ethernet IP for camera               ║"
        echo -e "║   12) Re-run full setup                  (needs net) ║"
        echo -e "║   13) Sync Python dependencies           (needs net) ║"
        echo -e "║   14) Autostart (systemd) — install / enable / logs  ║"
        echo -e "╠══════════════════════════════════════════════════════╣"
        echo -e "║    0) Exit                                           ║"
        echo -e "╚══════════════════════════════════════════════════════╝${N}"
        echo ""
        read -rp "  Select option: " OPT

        run_option "$OPT" || true
    done
}

# =============================================================================
# 7b. NON-INTERACTIVE DISPATCH  (callable from `./jetson.sh run <N>` or systemd)
# =============================================================================
# Every numbered menu item delegates here.  Splitting the dispatch out lets the
# headless autostart (systemd) and SSH-from-field workflow reuse the SAME code
# path as the interactive menu — no behaviour drift.
#
# SAFETY: option 6 (FC LIVE) prompts for a literal "fly" confirmation.  When
# called non-interactively (no TTY) it REFUSES to run — this prevents the
# Jetson from auto-arming and taking off on boot.
run_option() {
    local OPT="$1"
    case "$OPT" in
        1)
            run_pipeline --no-operator-view
            ;;
        2)
            MJPEG_IP="${JETSON_IP:-192.168.144.10}"
            echo -e "\n${G}MJPEG stream → open http://${MJPEG_IP}:8090 in a browser${N}\n"
            run_pipeline --operator-view-mode mjpeg
            ;;
        3)
            run_pipeline --operator-view-window-backend opencv
            ;;
        4)
            run_pipeline --no-operator-view \
                --no-gimbal-follow
            ;;
        5)
            # Pipeline + FC dry-run: computes MAVLink commands and logs
            # them to flight_commands.jsonl but does NOT open serial.
            # Safe to run on the ground / on the bench.  No internet needed.
            info "FC dry-run: commands logged only, serial NOT opened"
            run_pipeline --no-operator-view \
                --flight-control-enabled \
                --flight-control-dry-run
            ;;
        6)
            # Pipeline + FC LIVE: opens serial to ArduPilot and sends real
            # commands (GUIDED + arm + takeoff + yaw + alt-hold).
            # GUARDED — REFUSED in non-interactive mode so systemd cannot
            # accidentally arm the drone on boot.
            if [[ ! -t 0 ]]; then
                fail "Option 6 (FC LIVE) requires interactive confirmation. " \
                     "Refusing to run from a non-TTY context (systemd / ssh -T)."
            fi
            warn "FC LIVE will arm the FC and take off to flight_control.guided_alt_m"
            read -rp "  Type 'fly' to confirm real flight: " CONFIRM
            if [[ "$CONFIRM" != "fly" ]]; then
                info "Cancelled."
                return 0
            fi
            run_pipeline \
                --flight-control-enabled \
                --flight-control-live \
                --no-gimbal-follow
            ;;
        7)
            export_engine
            ;;
        8)
            run_preflight
            ;;
        9)
            run_smoke
            ;;
        10)
            verify_env
            check_engine
            ;;
        11)
            configure_network
            ;;
        12)
            do_setup
            ;;
        13)
            # Manual dependency sync — only run when you have internet.
            # Picks up new packages added to requirements-jetson.txt after
            # a git pull.  pip skips already-installed packages instantly.
            install_requirements
            ;;
        14)
            autostart_menu
            ;;
        0)
            echo -e "\n${G}Goodbye.${N}\n"
            exit 0
            ;;
        *)
            warn "Unknown option: $OPT"
            return 1
            ;;
    esac
}

# =============================================================================
# 7c. AUTOSTART (systemd user service)
# =============================================================================
# Installs ~/.config/systemd/user/skyscouter.service so the pipeline starts on
# boot.  Service runs `./jetson.sh run $SKYSCOUTER_BOOT_OPTION` with the chosen
# option (1-5 only; option 6 is refused above when no TTY).
#
# The model is picked via SKYSCOUTER_MODEL env var, also baked into the unit.
SERVICE_PATH="$HOME/.config/systemd/user/skyscouter.service"

autostart_write_unit() {
    local OPT="$1"
    local MODEL="$2"
    mkdir -p "$(dirname "$SERVICE_PATH")"
    cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=SkyScouter autonomous interceptor pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO
Environment=SKYSCOUTER_MODEL=$MODEL
Environment=SKYSCOUTER_BOOT_OPTION=$OPT
ExecStart=$REPO/jetson.sh run $OPT
Restart=on-failure
RestartSec=5
# Don't restart on Ctrl+C / clean exit — only restart on crashes
RestartPreventExitStatus=0 130 143

[Install]
WantedBy=default.target
EOF
    ok "Wrote $SERVICE_PATH"
    info "  Boot option: $OPT"
    info "  Model:       $MODEL"
}

autostart_menu() {
    hdr "Autostart (systemd user service)"
    echo "  Status: $(systemctl --user is-enabled skyscouter.service 2>/dev/null || echo 'not installed')"
    echo "          $(systemctl --user is-active  skyscouter.service 2>/dev/null || echo 'not running')"
    echo ""
    echo "  1) Install / update autostart (pick boot option + model)"
    echo "  2) Enable autostart on boot"
    echo "  3) Disable autostart on boot"
    echo "  4) Start now"
    echo "  5) Stop now"
    echo "  6) Show logs (journalctl -fu, follow)"
    echo "  0) Back"
    read -rp "  Select: " A
    case "$A" in
        1)
            echo ""
            echo "  Safe boot options (option 6 refused — would auto-takeoff):"
            echo "    1) Pipeline (no display)"
            echo "    2) Pipeline + MJPEG stream"
            echo "    3) Pipeline + OpenCV window"
            echo "    4) Pipeline (no gimbal)"
            echo "    5) Pipeline + FC dry-run"
            read -rp "  Boot option [1-5]: " BOPT
            if [[ ! "$BOPT" =~ ^[1-5]$ ]]; then
                warn "Refused: must be 1-5 (option 6 cannot autostart)"
                return 1
            fi
            select_model "pipeline"
            local BMODEL
            BMODEL=$(basename "$SELECTED_MODEL_DIR")
            autostart_write_unit "$BOPT" "$BMODEL"
            systemctl --user daemon-reload
            info "Run 'systemctl --user enable skyscouter' to start on boot."
            info "Or pick option 2 from this menu."
            ;;
        2)
            [ -f "$SERVICE_PATH" ] || fail "Install first (option 1)."
            systemctl --user enable skyscouter.service && ok "Enabled"
            # Make sure it runs even without an active login session
            loginctl enable-linger "$USER" 2>/dev/null && info "User lingering enabled"
            ;;
        3)
            systemctl --user disable skyscouter.service && ok "Disabled"
            ;;
        4)
            systemctl --user start skyscouter.service && ok "Started"
            ;;
        5)
            systemctl --user stop skyscouter.service && ok "Stopped"
            ;;
        6)
            journalctl --user -fu skyscouter.service
            ;;
        0|*)
            ;;
    esac
}

# =============================================================================
# ENTRY POINT
# =============================================================================
cd "$REPO"

case "${1:-}" in
    setup)
        do_setup
        ;;
    verify)
        verify_env
        check_engine
        ;;
    run)
        # Non-interactive dispatch: `./jetson.sh run <N>` runs menu option N
        # without prompting.  Used by SSH-from-field and the systemd autostart
        # service.  Model is picked via $SKYSCOUTER_MODEL (else falls back to
        # the interactive picker, which fails cleanly on a non-TTY).
        if [ -z "${2:-}" ]; then
            fail "Usage: ./jetson.sh run <option_number>"
        fi
        if ! venv_is_healthy; then
            fail "Venv not ready. Run: ./jetson.sh setup"
        fi
        run_option "$2"
        ;;
    autostart)
        # Shortcut: `./jetson.sh autostart` opens the systemd management menu.
        autostart_menu
        ;;
    *)
        if ! venv_is_healthy; then
            # Venv is missing, broken, or from a different machine — full setup.
            # This still needs the network on first install, but only on first
            # install.  After that, the entry point never touches the network.
            echo -e "${Y}Environment not ready — running full setup...${N}"
            do_setup
        fi
        # IMPORTANT: do NOT run install_requirements here on a warm venv.
        # In the field with no internet, that pip call hangs/errors and the
        # menu never appears, blocking access to the live pipeline.  Dependency
        # sync is now a menu option (11) for when you have connectivity.
        show_menu
        ;;
esac
