# DH116 physical WebRTC preview

This is an independent zero-PPO preview, not the native Teacher worker. Its
source-bound contract is created only after the display and RGB modules have
been integrated into the same checkout. No Isaac native execution or successful
WebRTC session is established by the CPU tests.

On the Ubuntu host, from the integrated checkout, create the contract outside
the repository:

```bash
/mnt/mydisk/xk/envs/dh116/bin/python scripts/preview_dh116_physical_webrtc.py \
  --freeze /absolute/path/preview-contract.json
```

The command prints the exact contract SHA-256. Review and separately authorize
that concrete contract before a native run. Run with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/xk/envs/dh116/bin/python \
  scripts/preview_dh116_physical_webrtc.py --execute-native \
  --contract /absolute/path/preview-contract.json \
  --contract-sha256 THE_PRINTED_SHA256 --output /absolute/path/new-output
```

The launcher selects private WebRTC mode, an explicit headless rendering
experience, and two 640x480 RGB cameras. It publishes the measured reset mesh,
then one owned mesh/RGB snapshot after each completed physical control
interval; a terminal partial interval is published only when it actually
produced a physical row. First frame is held for about 12 seconds to allow a
client to connect. Left/right RGB PNGs and per-frame identity JSON are written
to the output. Commands are fixed zero 24D actions; this entry does not provide
interactive joint control.

The bounded request comprises 2 setup SDK steps and at most 16 control
intervals of 8 SDK steps: 130 SDK steps total, including 128 measured SDK
steps and 1,536 solver substeps. The single shared budget allows at most 162
calls through SimulationContext.render, including reset warm-up, stereo
capture retries and 24 first-frame hold calls. Renderer activity internal to
Kit that bypasses this method is not measured by that counter. The 180-second
deadline is cooperative: a blocked SDK call is not interrupted, and no outer
watchdog or process termination is implemented. Journal physical rows flush
but are not individually fsynced; the existing journal makes its normal
completed/terminal boundaries durable.

The contract hashes repository Python sources, JSON configurations and JSON
contracts, plus the explicit Kit experience. The backend profile and cleanup
helper separately validate their pinned external sources and binary. This is
not a hash of the complete installed Isaac/Kit runtime. The existing singleton
lock is held across App creation and the fault-safe lifecycle. Native
viewport, camera, clock, resource-release and actual grasp behavior remain
unverified until an independently authorized native run is observed.
