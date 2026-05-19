# SIYI A8 Mini Jetson Integration

This is the production topology for using the SIYI A8 Mini as the main camera
for Jetson inference and the Insight 5G system as the live operator video
downlink.

## Topology

```text
SIYI A8 Mini Ethernet video
  -> Jetson RTSP ingest
      -> TensorRT detection/tracking/overlay
          -> Jetson fullscreen HDMI display
              -> Insight 5G transmitter
                  -> Insight 5G receiver
                      -> operator display
```

Telemetry/status remains separate:

```text
Jetson target-state/health messages
  -> USB telemetry radio
      -> ground telemetry radio
          -> operator laptop
```

## Camera Network

Default A8 Mini camera IP:

```text
192.168.144.25
```

Default RTSP URL for A8 Mini / older SIYI cameras:

```text
rtsp://192.168.144.25:8554/main.264
```

Some newer SIYI docs mention private stream ports for newer cameras, but A8 Mini
is documented as using the old RTSP address family.

The Jetson Ethernet interface must be on the same subnet, for example:

```text
Jetson IP: 192.168.144.10
Netmask:   255.255.255.0
Camera IP: 192.168.144.25
```

## Ready Launcher

Use:

```bash
./scripts/run_first_flight_siyi_a8_insight.sh
```

It does the following:

- pings the A8 Mini IP
- probes the RTSP stream
- runs the Stage 2 TensorRT engine
- records raw and annotated review video
- writes target state, diagnostics, and manifest artifacts
- shows live annotated output on the Jetson HDMI display through GStreamer
- feeds the HDMI display path into the Insight transmitter

## RTSP URL Override

If the camera is configured differently:

```bash
SIYI_A8_CAMERA_IP=192.168.144.26 \
SIYI_A8_RTSP_URL=rtsp://192.168.144.26:8554/main.264 \
./scripts/run_first_flight_siyi_a8_insight.sh
```

## Troubleshooting

If ping fails:

- confirm A8 Mini power
- confirm Ethernet wiring
- confirm Jetson Ethernet IP is in `192.168.144.0/24`
- confirm no laptop/router is using the same IP

If ping passes but RTSP fails:

- open the RTSP URL in VLC/QGroundControl/EasyPlayer from a machine on the same
  subnet
- try `rtsp://192.168.144.25:8554/main.264`
- confirm camera firmware and SIYI FPV camera stream settings

If the pipeline runs but Insight shows no image:

- verify Jetson HDMI display works directly on a monitor
- verify the Insight transmitter sees HDMI input
- keep the operator view backend as `gstreamer`; the Jetson venv OpenCV build is
  headless and cannot create HighGUI windows

## References

- SIYI A8 Mini User Manual v1.9: https://siyi.biz/siyi_file/A8%20mini/A8%20mini%20User%20Manual%20v1.9.pdf
- SIYI A8 Mini User Manual v1.6: https://siyi.biz/siyi_file/A8%20mini/A8%20mini%20User%20Manual%20v1.6.pdf
- ManualsLib A8 Mini v1.1 mirror: https://www.manualslib.com/manual/2930381/Siyi-A8-Mini.html
