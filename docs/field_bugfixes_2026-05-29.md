# Field Bug Fixes — 2026-05-29

Bugs found and fixed during live field testing with the Jetson + SIYI A8 Mini + Pixhawk6X.

---

## Bug 1 — Drone never took off (pipeline crashed before FC connected)

**What happened:**  
You typed 'fly', the pipeline started, but the drone never armed or took off.
The log showed it was stuck at the camera-opening step and crashed before
even reaching the flight controller.

**Why:**  
The SIYI camera (air-unit) takes longer to power on in the field than in the
office. The pipeline would time out waiting for the first camera frame (20 s),
crash, and never reach the FC connection step. The FC and camera startup are
sequential — no camera = no pipeline = no takeoff.

**Fix:**  
- First-frame timeout extended from 20 s → **60 s**, giving the camera time
  to boot in the field.
- Progress messages printed every 5 s while waiting, so the operator can see
  it's working and not frozen.
- Reconnect attempts are now logged so you can see the camera being retried.

---

## Bug 2 — Pipeline looked completely frozen at startup (field)

**What happened:**  
In the field, 9 out of 10 attempts the terminal would show:
```
Opening in BLOCKING MODE
NvMMLiteOpen : Block : BlockType = 279
NvMMLiteBlockCreate : Block : BlockType = 279
```
…and then nothing. Looked like a crash. Rebooting didn't help.

**Why:**  
Those lines are normal NVIDIA hardware decoder messages. The pipeline was
actually waiting for the camera to serve its first video frame. In the office
the camera is already warm (~0.2 s), so it "just works." In the field the
camera boots cold and takes 15–40 s. The code had no visible progress during
that wait, so it looked frozen.

**Fix:**  
Same as Bug 1 — longer timeout + visible heartbeat messages.

---

## Bug 3 — ARM storm: FC was being armed 7+ times, takeoff sent 4+ times

**What happened:**  
The flight log showed:
```
[flight] sent force-ARM ... (×7)
[flight] sent NAV_TAKEOFF #1 … #4
[flight] FC ACK NAV_TAKEOFF -> FAILED
```

**Why:**  
After the first `NAV_TAKEOFF` was sent, ArduPilot auto-disarmed the drone
(because on the ground it couldn't detect a liftoff). The code saw "not armed"
and kept force-arming repeatedly. After 5 s it would send another `NAV_TAKEOFF`.
This looped until something else broke.

**Fix:**  
- Once `NAV_TAKEOFF` has been sent, the code no longer force-re-arms. ArduPilot
  owns the takeoff sequence from that point.
- Hard cap of **3 NAV_TAKEOFF attempts** (configurable). After 3 tries without
  altitude confirmation, the code prints "Halting takeoff sequence — verify the
  drone is able to fly" and stops.

---

## Bug 4 — Camera dropout crashed the pipeline mid-flight

**What happened:**  
The pipeline ran for ~22 seconds then crashed with:
```
RTSP stream 'siyi_a8_mini_main_rtsp' produced no new frames for 5.0s
```
The drone was left in an armed/flying state with no guidance.

**Why:**  
The GStreamer video decoder briefly dropped the stream (HEVC decode error or
small network glitch). The decoder needed up to 11 s to reconnect, but the
pipeline's tolerance was only 5 s — so it crashed before the reconnect
finished.

**Fix:**  
Read timeout increased from **5 s → 20 s**, giving the decoder enough time to
reconnect after a brief dropout without killing the pipeline.

---

## Bug 5 — No way to check readiness before flying

**What happened:**  
You'd type 'fly', the pipeline would start, and only then discover the camera
was unreachable or the FC wasn't connected — wasting time and risking a bad
state.

**Fix:**  
Pre-flight checks now run automatically before the 'fly' prompt:
- ✓ FC serial port exists and can be opened
- ✓ Camera network host is reachable (TCP ping)

Menu option **8 (Preflight check)** runs a full check including:
- MAVLink heartbeat from the FC (passive — **never arms, never spins motors**)
- RTSP camera stream probe
- Prints **GO / NO-GO** at the end

---

## Bug 6 — Pipeline ran for 60 s with no FC, doing nothing

**What happened:**  
After unplugging and replugging the FC USB, the pipeline started but logged:
```
FC: not connected (MAVLink connect failed: No such file or directory: '/dev/ttyACM0')
```
…and then kept running, detecting and tracking, but the drone never armed
or took off. No error, no crash. You had no idea it wasn't going to fly.

**Why:**  
The code built the FC link, `start()` failed silently, and the pipeline
continued anyway as if everything was fine.

**Fix:**  
In **LIVE mode**, if the FC fails to connect, the pipeline now immediately
aborts with a clear error:
```
FC LIVE mode: could not connect to flight controller — check USB cable…
```

---

## Bug 7 — Unplugging and replugging the FC USB broke the connection permanently

**What happened:**  
After unplugging and replugging the Pixhawk USB (or changing the USB port),
the pipeline could no longer find the FC even though it was clearly connected.
Changing cables and rebooting didn't always help.

**Why:**  
Linux assigns USB serial devices dynamically as `/dev/ttyACM0`, `/dev/ttyACM1`,
etc. After a replug, if the old `ttyACM0` entry was still registered, the new
connection appeared as `ttyACM1`. The code was hardcoded to `/dev/ttyACM0` and
couldn't find it.

**Fix:**  
1. **Permanent udev rule** (`/etc/udev/rules.d/99-pixhawk.rules`) creates a
   stable symlink `/dev/pixhawk` that always points to the Pixhawk's main
   MAVLink port, no matter which ACM number Linux assigns.  
   This was a **one-time setup** — it now works automatically forever, on every
   plug/replug/reboot.
2. Config updated to use `/dev/pixhawk` instead of `/dev/ttyACM0`.
3. At connect time, the code waits up to 5 s for USB enumeration (handles
   "port not ready yet" race), then falls back: `pixhawk → ttyACM0 → scan
   all ttyACM*`.

---

## Bug 8 — Takeoff altitude was fixed at 2 m, no way to change it

**What happened:**  
Every flight always took off to 2 m. To fly higher you had to edit a YAML
config file before each run.

**Fix:**  
The 'fly' prompt now asks for altitude before confirming:
```
Select takeoff altitude (AGL):
  1)   2 m   — close-range test  (default)
  2)   5 m
  3)  10 m
  4)  Custom
Choice [1]:
```
The chosen altitude is passed directly to the pipeline — no config editing
needed.

---

## Summary table

| # | Symptom | Root cause | Status |
|---|---------|-----------|--------|
| 1 | Drone never took off | Camera timeout too short (20 s) | ✅ Fixed (60 s + progress) |
| 2 | Terminal frozen at startup (field) | No visible wait for camera | ✅ Fixed (heartbeat every 5 s) |
| 3 | ARM sent 7× / TAKEOFF sent 4× | Re-arm after NAV_TAKEOFF auto-disarm | ✅ Fixed (no re-arm + 3-attempt cap) |
| 4 | Pipeline crash mid-flight | RTSP read timeout too tight (5 s) | ✅ Fixed (20 s) |
| 5 | No readiness check before flying | No preflight gate | ✅ Fixed (preflight + GO/NO-GO) |
| 6 | 60 s run with no FC, no error | Silent connect failure in LIVE mode | ✅ Fixed (hard abort) |
| 7 | USB replug breaks FC connection | ttyACM number changed dynamically | ✅ Fixed (/dev/pixhawk udev symlink) |
| 8 | Altitude always 2 m | Hardcoded in config | ✅ Fixed (interactive prompt) |
