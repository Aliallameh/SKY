#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/office/SKY"

# Dedicated launcher for the Insight 5G 1080p HDMI digital video downlink.
# It keeps the normal first-flight recording behavior, but shows the live
# annotated operator view fullscreen on the Jetson display output.
export OPERATOR_VIEW_MODE="window"
export OPERATOR_VIEW_FULLSCREEN="1"
export OPERATOR_VIEW_DISPLAY_WIDTH="1920"
export OPERATOR_VIEW_DISPLAY_HEIGHT="1080"
export OPERATOR_VIEW_WINDOW_BACKEND="gstreamer"
export OPERATOR_VIEW_DISPLAY_FPS="10"

exec "$ROOT/scripts/run_first_flight_tensorrt.sh"
