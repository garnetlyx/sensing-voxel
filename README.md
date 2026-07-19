# Sensing Voxel

Sensing Voxel is an experimental, safety-gated control and monitoring service for 3D printers. SVA (Sensing-Voxel Agent) provides Telegram and web-dashboard interfaces, LLM-assisted operations, deterministic G-code authorization, and camera-based print monitoring.

## Safety Disclaimer

Sensing Voxel combines automation, camera interpretation, printer-control APIs, and physical machinery. These systems can fail independently or together. The software reduces some avoidable risks but cannot prove that a printer, toolhead, build plate, material, environment, or generated motion is safe.

Operate printers with appropriate supervision, fire precautions, verified firmware limits, and a reachable hardware emergency stop. A successful software validation or camera check is not a safety certification. By using this project, you accept responsibility for machine behavior, print outcomes, hardware damage, material loss, fire risk, and other consequences.

## Current Capabilities

- **Unified print authorization**: One `PrintSafetyService` is created per configured printer and acts as the final gate for safety-managed print starts.
- **Whole-file semantic G-code validation**: Every executable line is interpreted in machine-state order. The validator applies command and parameter allowlists, tracks coordinate/extrusion/temperature state, checks motion and thermal semantics, calculates the toolpath and first-layer envelopes, and rejects unsupported or unsafe input.
- **Content-bound approval**: Source bytes are stored by SHA-256. Validation, deterministic preview artifacts, operator approval, printer identity, configuration state, and execution are cryptographically bound so changed inputs require a new decision.
- **Rendered preview and physical confirmation**: The approval flow provides deterministic JSON plus an SVG toolpath preview. Its evidence includes material and print metadata plus a physical snapshot derived from the configured build-plate memo, hotend/nozzle configuration, and live Z-offset state for operator review.
- **Live preflight**: Immediately before execution, the safety service rechecks mutable printer facts, including target identity, hardware/profile/config/macro digests, probe Z offset, homing-origin Z, source trust, and user identity. A mismatch fails closed.
- **Mandatory monitoring contract**: Prints whose source is `user_gcode` or `sva_generated` cannot start through the safety service unless `PrintMonitor` passes readiness checks and starts with a filename, source identity, and first-layer Z.
- **First-layer monitoring cadence**: Required jobs receive a camera snapshot and independent LLM image analysis every 60 seconds until first-layer completion; monitoring then uses the configured normal interval. An unacknowledged detected anomaly causes an automatic pause.
- **Klipper/Moonraker control**: The production runtime supports printer status, telemetry, upload/start, lifecycle commands, macros, and WebSocket updates through the Klipper adapter.
- **Telegram and dashboard interfaces**: User-allowlisted Telegram operation, REST/WebSocket dashboard updates, chat, approvals, audit information, and camera access.
- **Dashboard/API hardening**: `/health` is the only public path. Other HTTP routes require bearer authentication, missing authentication fails closed, non-loopback binding requires an API token, and WebSocket connections enforce configured token and origin checks.
- **Camera and PTZ support**: RTSP snapshots/streaming and ONVIF controls for supported Tapo cameras.
- **Printer memos and auditability**: Physical attributes that do not belong in printer firmware configuration can be stored in SQLite; tool actions and decisions are auditable.

## Safety-Managed G-code Flow

```text
G-code bytes
  -> content-addressed artifact
  -> whole-file deterministic validation
  -> deterministic JSON + rendered SVG preview
  -> operator approval + physical confirmation
  -> approval binding for artifact, validation, printer, config, and user
  -> first live printer/Z preflight
  -> mandatory PrintMonitor readiness and startup
  -> second live printer/Z preflight
  -> start the exact printer-side file through the selected adapter
  -> 60-second first-layer checks, then configured cadence
```

Any failed validation, stale binding, changed live fact, absent required confirmation evidence, adapter error, monitor-readiness failure, or failure to start mandatory monitoring prevents authorization or fails the start closed.

## Architecture

```text
Telegram / Web Dashboard
           |
       Agent Core
  (state, policy, decisions, LLM tools)
           |
  PrintSafetyService per printer
   |       |        |       |
artifact validator preview preflight
   |                         |
   +---- Printer Adapter ----+
              |
       Klipper/Moonraker
              |
         PrintMonitor
   (camera snapshot + isolated
      LLM anomaly analysis)
```

The deterministic authorization path and the probabilistic camera-monitoring path are deliberately separate. LLM output does not replace G-code validation or live preflight.

## Quick Start

### Prerequisites

- Python 3.12+
- A configured Klipper/Moonraker printer for the production-supported path
- Docker, optionally, for containerized deployment
- An RTSP camera and supported LLM provider for camera monitoring

### Installation

```bash
git clone https://github.com/garnetlyx/sensing-voxel.git
cd sensing-voxel
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp config/default.yaml config/local.yaml
# Edit config/local.yaml for your printer, interfaces, authentication, and camera.
python -m sensing_voxel.main
```

### Docker Compose

```bash
git clone https://github.com/garnetlyx/sensing-voxel.git
cd sensing-voxel
cp .env.example .env
# Configure required credentials and an API token for any non-loopback dashboard bind.
docker compose up -d
docker compose logs -f
```

Do not expose the dashboard to a non-loopback interface without a strong API token and appropriate network controls.

## Project Structure

```text
src/sensing_voxel/
├── main.py                  # Runtime composition and lifecycle
├── config.py                # Configuration models and loading
├── core/                    # State machine, decisions, persistence
├── safety/                  # Artifacts, validator, preview, authorization, execution
├── monitoring/
│   └── print_monitor.py     # Required camera/LLM print monitor
├── adapters/                # Printer adapter contract and implementations
├── bot/                     # Telegram interface and authorization
├── api/                     # Hardened HTTP/WebSocket dashboard API
├── dashboard/               # Browser UI assets
├── llm/                     # Chat, tools, approvals, context, and audit
├── camera/                  # ONVIF PTZ support
├── slicer/                  # Model processing support; not fully runtime-wired
├── storage/                 # Runtime persistence used by current services
└── infra/                   # Shared infrastructure utilities
```

## Current Limitations

- Camera anomaly assessment is probabilistic. Lighting, occlusion, focus, camera movement, network faults, and model error can produce false positives or false negatives.
- The monitor pauses on an unacknowledged anomaly; it does not perform automatic first-layer Z-offset correction.
- The mandatory first-layer check interval is 60 seconds, so this is not sub-minute hazard detection or a replacement for firmware and hardware protection.
- Deterministic validation covers the supported G-code dialect and modeled machine semantics only. Unsupported commands or parameters are rejected rather than guessed.
- Preview rendering is bounded to 20,000 extrusion segments and 2,000,000 SVG bytes. It is an approval aid, not a physical simulation or collision proof.
- The content-addressed artifact store used by the safety service is in memory; authorization artifacts do not survive a process restart unless persisted elsewhere by a surrounding workflow.
- Physical confirmation presents configured and live evidence for operator approval. The software cannot independently verify that the bed is clear, material is correct, hardware is intact, or the environment is supervised.
- The live preflight detects changes in facts it can query and bind; it cannot detect every mechanical, electrical, firmware, or environmental hazard.
- Klipper/Moonraker is the production-supported printer path. The Bambu adapter exists but is not production-wired by the default runtime configuration.
- Model slicing and photo-to-print generation are not current end-to-end runtime workflows. Supply already generated G-code to use the complete safety-managed path.
- Dashboard security protects the application boundary but does not provide TLS termination, host firewalling, VPN access, credential rotation, or network segmentation.

## Development

```bash
pytest
pytest --cov=sensing_voxel --cov-report=html
```

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).

## Credits

Inspired by patterns from [OpenClaw](https://github.com/openclaw/openclaw).
