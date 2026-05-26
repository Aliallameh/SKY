#!/bin/bash
# =============================================================================
# SkyScouter — Jetson Setup & Launcher
# =============================================================================
# Single entry point for a fresh Jetson Orin.  Run once to set up the
# environment, then use the menu to run any pipeline or tool.
#
# Usage:
#   chmod +x jetson.sh
#   ./jetson.sh          # setup (if needed) then show menu
#   ./jetson.sh setup    # force full environment setup
#   ./jetson.sh verify   # verify environment only
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

    # requirements-jetson.txt specifies opencv-python-headless (no GUI).
    # We need the GUI-capable build for the operator view window.
    # Install it AFTER requirements so it replaces headless if pip installed it.
    # Do NOT rely on system python3-opencv — it was compiled against NumPy 1.x
    # and fails with NumPy 2.x in this venv.
    info "Installing opencv-python (GUI-capable, replaces headless)..."
    piprun install "opencv-python>=4.7,<5"
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
    CFG_ENGINE=$(grep "weights:" "$REPO/configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml" \
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
# 5. ENGINE EXPORT
# =============================================================================
export_engine() {
    hdr "Export TensorRT engine"
    echo "Available model directories:"
    ls -d "$REPO/data/models"/yolov26_lrdd_v2*/ 2>/dev/null | nl -ba || echo "  (none found)"
    echo ""
    read -rp "Model directory path: " MODEL_DIR
    MODEL_DIR="${MODEL_DIR%/}"
    PT="$MODEL_DIR/best.pt"
    if [ ! -f "$PT" ]; then
        fail "best.pt not found at $PT"
    fi
    FILE_SIZE=$(stat -c%s "$PT")
    if [ "$FILE_SIZE" -lt 1000 ]; then
        fail "best.pt is only $FILE_SIZE bytes — it is a git-LFS pointer, not real weights."$'\n'"       Run: sudo apt-get install git-lfs -y && git lfs install && git lfs pull"
    fi
    read -rp "Input image size [1024]: " IMGSZ
    IMGSZ="${IMGSZ:-1024}"
    read -rp "Precision — FP16 (faster, recommended) or FP32? [fp16]: " PREC
    PREC="${PREC:-fp16}"
    echo ""
    HALF_FLAG=""
    if [[ "${PREC,,}" == "fp16" || "${PREC,,}" == "half" ]]; then
        HALF_FLAG="--half"
        info "Exporting FP16 TRT engine (input ${IMGSZ}×${IMGSZ}) — takes ~6 minutes..."
    else
        info "Exporting FP32 TRT engine (input ${IMGSZ}×${IMGSZ}) — takes ~4 minutes..."
    fi
    pyrun "$REPO/scripts/export_tensorrt.py" \
        --weights "$PT" \
        --imgsz "$IMGSZ" \
        $HALF_FLAG
    ok "Engine exported to $MODEL_DIR/best.engine"
    echo ""
    read -rp "Set this engine as the active config? [Y/n]: " ANS
    ANS="${ANS:-Y}"
    if [[ "$ANS" =~ ^[Yy] ]]; then
        RELPATH="${MODEL_DIR#$REPO/}/best.engine"
        sed -i "s|weights:.*best.engine|weights: \"$RELPATH\"|" \
            "$REPO/configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml"
        sed -i "s|input_size:.*|input_size: $IMGSZ|" \
            "$REPO/configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml"
        ok "Config updated → $RELPATH @ imgsz=$IMGSZ"
    fi
}

# =============================================================================
# 6. PIPELINE LAUNCHERS
# =============================================================================
default_output() {
    # Usage: default_output <tag>   e.g.  default_output "live"
    echo "$REPO/data/outputs/${1}_$(date -u +%Y%m%dT%H%M%SZ)"
}

run_pipeline() {
    local EXTRA_ARGS=("$@")
    local OUTDIR
    OUTDIR=$(default_output "jetson_live")
    info "Output dir → $OUTDIR"
    cd "$REPO"
    pyrun scripts/run_pipeline.py \
        --config configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml \
        --output "$OUTDIR" \
        "${EXTRA_ARGS[@]}"
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
    info "Running pipeline in log-only mode (gimbal + display disabled). Ctrl+C to stop."
    local OUTDIR
    OUTDIR=$(default_output "smoke")
    cd "$REPO"
    pyrun scripts/run_pipeline.py \
        --config configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml \
        --output "$OUTDIR" \
        --no-operator-view \
        --no-gimbal-follow
}

get_jetson_ip() {
    ip addr show eno1 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1 | head -1
}

# =============================================================================
# 7. MENU
# =============================================================================
show_menu() {
    while true; do
        JETSON_IP=$(get_jetson_ip)
        ENGINE=$(grep "weights:" "$REPO/configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml" \
                 | head -1 | awk '{print $2}' | tr -d '"' | xargs basename 2>/dev/null)
        IMGSZ=$(grep "input_size:" "$REPO/configs/deploy_jetson_yolov26_lrdd_v2_siyi_a8_mini_1080p.yaml" \
                | head -1 | awk '{print $2}')

        echo -e "\n${C}╔══════════════════════════════════════════════════════╗"
        echo -e "║            SkyScouter — Jetson Launcher              ║"
        echo -e "╠══════════════════════════════════════════════════════╣"
        printf  "║  Engine : %-43s║\n" "$ENGINE  (imgsz=$IMGSZ)"
        printf  "║  Jetson : %-43s║\n" "${JETSON_IP:-not configured}  →  Camera: 192.168.144.25"
        echo -e "╠══════════════════════════════════════════════════════╣"
        echo -e "║  PIPELINE                                            ║"
        echo -e "║    1) Run live pipeline  (TRT — no display)          ║"
        echo -e "║    2) Run live pipeline  (TRT — MJPEG stream)        ║"
        echo -e "║    3) Run live pipeline  (TRT — OpenCV window)       ║"
        echo -e "║    4) Run live pipeline  (gimbal follow DISABLED)    ║"
        echo -e "╠══════════════════════════════════════════════════════╣"
        echo -e "║  TOOLS                                               ║"
        echo -e "║    5) Export TRT engine from .pt weights             ║"
        echo -e "║    6) Preflight check  (camera + deps + config)      ║"
        echo -e "║    7) Smoke test  (30-second run)                    ║"
        echo -e "║    8) Verify environment                             ║"
        echo -e "║    9) Configure Ethernet IP for camera               ║"
        echo -e "║   10) Re-run full setup                              ║"
        echo -e "╠══════════════════════════════════════════════════════╣"
        echo -e "║    0) Exit                                           ║"
        echo -e "╚══════════════════════════════════════════════════════╝${N}"
        echo ""
        read -rp "  Select option: " OPT

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
                export_engine
                ;;
            6)
                run_preflight
                ;;
            7)
                run_smoke
                ;;
            8)
                verify_env
                check_engine
                ;;
            9)
                configure_network
                ;;
            10)
                do_setup
                ;;
            0)
                echo -e "\n${G}Goodbye.${N}\n"
                exit 0
                ;;
            *)
                warn "Unknown option: $OPT"
                ;;
        esac
    done
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
    *)
        # Auto-setup if the venv is missing or stale, then show menu
        if ! venv_is_healthy; then
            echo -e "${Y}Environment not ready — running setup first...${N}"
            do_setup
        fi
        show_menu
        ;;
esac
