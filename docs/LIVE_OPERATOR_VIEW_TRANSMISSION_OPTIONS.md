# Live Operator View Transmission Options

This document lists practical low-cost ways to see the Jetson's live annotated
model output during a flight or bench run. It focuses on topology and hardware,
not software commands.

## Goal

Show the operator what the Jetson model is doing live:

- camera image
- detections
- tracking boxes
- lock/tracking state
- confidence and latency overlays

The current USB telemetry radio is useful for low-bandwidth model state, but it
is not a video link.

## Current Hardware Constraint

The available "FPV radio telemetry ground module" is assumed to be a USB serial
telemetry radio in the SiK/3DR/Holybro/RFD style.

Topology:

```text
Jetson USB
  -> telemetry radio air module
      -> RF telemetry link
          -> telemetry radio ground module
              -> operator laptop USB serial
```

Use this for:

- detection count
- class label
- confidence
- bounding box center and size
- track id
- lock state
- FPS
- pipeline latency
- health/status messages

Do not use this for:

- live MJPEG video
- H.264/H.265 video
- raw frames
- browser video stream

Reason: telemetry radios are generally tens to hundreds of kbps. Live video
needs hundreds of kbps to multiple Mbps even after compression.

## Recommended Options

## Option 1: Use Existing Telemetry Radio For Metadata Only

Best when the immediate goal is to prove that detection/tracking is happening
live, without seeing the image.

Topology:

```text
Camera
  -> Jetson detection/tracking
      -> compact target-state messages
          -> USB telemetry radio
              -> RF link
                  -> ground telemetry radio
                      -> operator laptop dashboard/log viewer
```

Suggested modules:

- Existing FPV telemetry radio module
- SiK telemetry radio pair
- 3DR-style telemetry radio pair
- Holybro SiK Telemetry Radio V3
- RFD900/RFD900x class radios if already available

Required adapters:

- USB cable from Jetson to airborne telemetry module
- USB cable from ground telemetry module to laptop
- Proper antennas for the selected telemetry band
- Stable 5V supply if the airborne module cannot be powered safely from Jetson USB

Pros:

- Cheapest path because the hardware is already available
- Long range compared with normal Wi-Fi
- Good for live state, debugging, and logging
- Low integration risk

Cons:

- No visual frame
- No image overlay
- Operator must infer behavior from numbers/state

Use this as the fallback/status link even if a separate video link is added.

## Option 2: Dedicated Video Downlink From Jetson HDMI

Best way to see an actual live annotated image without building an IP network.
This can be either a dedicated digital HDMI video system or a cheaper analog FPV
path.

Topology:

```text
Camera
  -> Jetson detection/tracking
      -> annotated HDMI output
          -> HDMI digital video transmitter or HDMI-to-AV converter
              -> video RF link
                  -> receiver
                      -> phone/tablet/PC app, monitor, goggles, or USB capture
```

Suggested modules:

- Insight 5G 1080P 100mW Full HD Digital Video Transmission System
- Skyzone TX-5D HDMI/Analog 5.8 GHz VTX
- Generic 5.8 GHz analog FPV VTX, such as AKK/Eachine/TS5828-style modules
- Generic 5.8 GHz analog FPV receiver, such as RC832-style receiver
- USB UVC FPV receiver, such as Eachine ROTG02, for laptop/Android viewing

Required adapters:

- Jetson display-output cable
- Active DisplayPort-to-HDMI adapter if the Jetson carrier exposes DisplayPort
  instead of HDMI
- Full-size HDMI to Micro HDMI cable or adapter for the Insight transmitter
- HDMI-to-AV/composite converter only if using a normal analog VTX
- Video transmitter antennas
- Video receiver antennas
- Ground display, goggles, receiver app/device, or USB capture receiver
- Airborne power regulator for the transmitter
- Optional HDMI dummy plug if the Jetson needs a forced display mode

Pros:

- Uses the Jetson's real annotated display output
- Low latency
- Does not require an IP network
- Works with simple ground displays
- The Insight 5G system is a strong fit if it is already available because it
  accepts HDMI/Micro HDMI video, advertises 1080p-class digital transmission,
  HEVC compression, 5.1-5.8 GHz operation, MIMO-OFDM, and about 80 ms delay

Cons:

- It does not provide normal IP access to the Jetson
- It does not replace SSH/log transfer
- Receiver/app compatibility must be tested before flight
- HDMI display mode and scaling must be configured so the overlay is readable
- Analog variants have limited image quality, and small overlay text can be hard
  to read
- HDMI-to-AV conversion adds wiring and failure points for analog systems
- Ground laptop access to Jetson logs/SSH is not provided
- RF channel/power legality must be checked for the flight location

This is the most practical option if the operator only needs to see the
annotated image and does not need network access to the Jetson. Because the
Insight kit is already available, it should be tested before buying an analog
VTX or an IP bridge.

## Option 3: Cheap Point-To-Point IP Bridge

Best match for the current Jetson browser-based live operator view.

Topology:

```text
Camera
  -> Jetson detection/tracking
      -> annotated MJPEG or future H.264 stream
          -> Jetson Ethernet or USB-Ethernet adapter
              -> airborne IP bridge radio
                  -> air-to-ground IP link
                      -> ground IP bridge radio
                          -> operator laptop browser/VLC/SSH
```

Suggested modules:

- TP-Link CPE210, 2.4 GHz outdoor CPE
- TP-Link CPE510, 5 GHz outdoor CPE
- Ubiquiti NanoStation 5AC Loco
- Similar outdoor point-to-point bridge kits with Ethernet and PoE

Required adapters:

- USB-Ethernet adapter for Jetson if the onboard Ethernet port is unavailable
- Short Ethernet cable from Jetson to airborne bridge
- PoE injector or DC PoE splitter compatible with the bridge
- Airborne DC-DC regulator sized for the bridge power draw
- Ground PoE injector/power supply
- Ethernet cable from ground bridge to laptop or router
- Mounts to keep antennas reasonably aimed

Pros:

- Works with normal IP traffic
- Supports live browser view
- Can also carry SSH, logs, telemetry, and file transfer
- Better image quality than analog if the RF link is stable

Cons:

- Directional antennas are awkward on a moving drone
- More payload, power, and mounting work
- Requires IP configuration
- Line of sight and antenna orientation matter
- 2.4 GHz may conflict with RC systems; 5 GHz has less penetration

This is the best low-cost solution if the operator laptop must access the
Jetson directly.

## Option 4: Short-Range USB Wi-Fi Or Router Link

Best for bench tests, taxi tests, and very short hover tests near the operator.

Topology:

```text
Camera
  -> Jetson detection/tracking
      -> annotated browser stream
          -> Jetson Wi-Fi or travel router
              -> short-range Wi-Fi
                  -> operator laptop browser
```

Suggested modules:

- Jetson-compatible USB Wi-Fi adapter
- Small travel router
- Existing field router or laptop hotspot
- ALFA AWUS036ACH/AWUS036ACM class adapter for stronger bench/field Wi-Fi

Required adapters:

- USB Wi-Fi adapter or USB-Ethernet adapter
- Powered USB hub if the adapter draws too much power
- Small 5V regulator for travel router if used onboard

Pros:

- Cheap
- Fast to test
- Uses current browser operator view
- Good for development and close-range validation

Cons:

- Not a reliable flight-range solution
- Interference-sensitive
- Range and stability are unpredictable
- Should not be treated as a safety-critical link

Use this before buying larger hardware, but do not rely on it for meaningful
flight distance.

## Option 5: OpenHD-Style Digital FPV Link

Best if the team is willing to build a dedicated open-source digital video
system.

Topology:

```text
Camera / Jetson video output
  -> air-side OpenHD-compatible computer/radio path
      -> supported Wi-Fi adapter link
          -> ground-side OpenHD-compatible computer/radio path
              -> QOpenHD display or ground laptop
```

Suggested modules:

- OpenHD-supported SBC for air/ground side
- Raspberry Pi, Rock5, X86, or current OpenHD-supported hardware
- Supported Wi-Fi adapters using chipsets such as RTL8812AU/RTL8814AU/RTL8812EU
- ALFA AWUS036ACH
- ASUS USB-AC56
- ALFA AWUS1900

Required adapters:

- HDMI capture/CSI-HDMI adapter if feeding Jetson HDMI into OpenHD air side
- Supported Wi-Fi adapters and antennas
- Proper power supplies for air and ground computers
- Ground display or laptop

Pros:

- Designed for UAV digital video
- Can combine video and telemetry in one system
- Better digital image quality than analog
- Open-source ecosystem

Cons:

- More complex than an IP bridge or analog VTX
- Hardware compatibility matters
- Jetson support is not the simplest path
- Additional SBCs/adapters increase payload and integration work

This is not the fastest MVP path, but it is worth considering if the project
needs a reusable digital FPV stack.

## Practical Ranking

For the current project, the recommended order is:

1. Existing Insight 5G HDMI digital video system for live annotated image.
2. Existing telemetry radio for live model metadata.
3. Analog FPV from Jetson HDMI if the Insight system fails or is unavailable.
4. Point-to-point IP bridge if the operator needs browser video plus SSH/logs.
5. Short-range Wi-Fi only for bench and close hover tests.
6. OpenHD only if the team accepts extra integration complexity.

## Minimum Purchase Lists

## Existing Insight 5G Visual Setup

- Insight 5G 1080P 100mW transmitter
- Insight 5G receiver
- Transmitter antennas
- Receiver antennas
- Jetson HDMI or DisplayPort-to-HDMI output path
- Full-size HDMI to Micro HDMI cable or adapter into the transmitter
- Airborne 5-18V power feed sized for about 8W transmitter draw
- Ground receiver battery, USB power, or power bank
- Phone/tablet/PC receiver app or compatible ground display path
- Mounting hardware and strain relief for HDMI, power, and antennas

## Cheapest Visual Setup

- HDMI-to-AV converter
- 5.8 GHz analog VTX
- 5.8 GHz analog receiver or UVC FPV receiver
- VTX/RX antennas
- Airborne power regulator
- Ground display or laptop capture receiver

## Cheapest IP Setup

- Two TP-Link CPE210/CPE510 units or two Ubiquiti NanoStation 5AC Loco units
- USB-Ethernet adapter for Jetson if needed
- Airborne PoE/DC power solution
- Ground PoE injector/power supply
- Ethernet cables
- Basic mounting hardware

## Telemetry-Only Setup

- Existing USB telemetry radio pair
- Proper antennas
- USB cables
- Optional 5V power regulator
- Ground laptop serial viewer/dashboard

## Recommended MVP Decision

If the first flight only needs the operator to visually confirm detection and
tracking, use the existing Insight 5G HDMI digital video system from the Jetson
HDMI output first.

If the first flight also needs SSH, logs, and browser access to the Jetson,
choose a point-to-point IP bridge pair.

Keep the existing USB telemetry radio for low-bandwidth lock/tracking state in
both cases.

## References Checked

- TP-Link CPE210: https://www.tp-link.com/us/business-networking/outdoor-radio/cpe210/
- TP-Link CPE510: https://www.tp-link.com/us/business-networking/outdoor-radio/cpe510/v3/
- Ubiquiti NanoStation 5AC Loco: https://store.ui.com/us/en/category/all-wireless/products/loco5ac
- Insight 5G 1080P system: https://stockrc.com/en/components/783-insight-5g-1080p-100mw-full-hd-digital-video-transmission-system.html
- Insight 5G 1080P system alternate listing: https://tienda.stockrc.com/Insight-5G-1080P-100mW-Full-HD-Digital-Video-Transmission-System/en
- Skyzone TX-5D HDMI/Analog VTX: https://www.getfpv.com/skyzone-tx-5d-5-8ghz-32ch-600mw-av-transmitter-w-hdmi-port.html
- Eachine ROTG02 UVC FPV receiver: https://www.eachine.com/Eachine-ROTG02-UVC-OTG-5_8G-150CH-Dual-Antenna-Audio-FPV-Receiver-for-Android-Tablet-Smartphone-p-1063.html
- OpenHD hardware overview: https://openhdfpv.org/hardware/sbcs
- OpenHD Wi-Fi adapters: https://openhdfpv.org/hardware/wifi-adapters/
