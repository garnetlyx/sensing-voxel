# Sensing Voxel

AI-powered 3D printer monitoring and control system with real-time vision-based quality detection.

> **SVA** (Sensing-Voxel Agent) is the AI agent runtime at the core — it drives the LLM tool-use loop, state machine, and all wired subsystems. Interact with SVA via the Telegram bot or web dashboard chat.

## Safety Disclaimer

Sensing Voxel is experimental software. It combines LLM-driven automation, camera-based interpretation, printer control APIs, and physical 3D printers, all of which can behave unpredictably in real-world conditions.

The author has made best efforts to design safer workflows, approval gates, validation checks, and conservative defaults, but no software layer can predict every printer state, hardware failure, model error, configuration mistake, network issue, or unsafe operating condition. Use this project only with appropriate supervision and at your own risk.

By using Sensing Voxel, you accept full responsibility for any print outcome, machine behavior, hardware damage, material loss, fire risk, or other consequence. The author is not responsible for damage to printers, tools, property, or any other losses arising from use, modification, or deployment of this software.

## Features

- 🤖 **Telegram Bot Interface**: Upload G-code files, receive notifications, and control prints
- 🖨️ **Multi-Printer Support**: Klipper/Moonraker (Bambu adapter exists but is not production-wired)
- 📊 **Web Dashboard**: Real-time monitoring UI with WebSocket updates
- 🔧 **First-Layer Auto-Correction**: Z-offset adjustment *(module-complete, not yet wired — see Roadmap)*
- 📸 **Camera Calibration**: Checkerboard-based bed-level calibration *(module-complete, not yet wired — see Roadmap)*
- 🛡️ **External Print Detection**: Monitor prints started outside the agent *(module-complete, not yet wired — see Roadmap)*

## Architecture

```
Telegram Bot <-> Agent Core (state machine + policy + decision engine)
                      |
      +---------------+-------------------+
      |               |                   |
 Slicer Pipeline   Vision Monitor     Printer Adapter (abstract)
 (PrusaSlicer CLI) (RTSP → OpenCV)     ├── KlipperAdapter (Moonraker REST/WS)
                   ├── FirstLayer       └── BambuAdapter (Local MQTT + FTP)
                   │   (OpenCV)
                   └── Calibration
                       (Checkerboard)
```

## Quick Start

### Prerequisites

- Python 3.12+
- Docker (optional, for containerized deployment)
- PrusaSlicer CLI (for slicing STL files)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/garnetlyx/sensing-voxel.git
cd sensing-voxel
```

2. Create virtual environment and install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. Configure:

```bash
cp config/default.yaml config/local.yaml
# Edit config/local.yaml with your settings
```

4. Run:

```bash
source .venv/bin/activate
python -m sensing_voxel.main
```

### Docker Deployment (One-Liner)

The fastest way to deploy Sensing Voxel is using Docker Compose:

```bash
# 1. Clone the repository
git clone https://github.com/garnetlyx/sensing-voxel.git
cd sensing-voxel

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env with your settings (at minimum, set SV_TELEGRAM_BOT_TOKEN)

# 3. Start the services
docker compose up -d

# 4. View logs
docker compose logs -f
```

**Configuration via environment variables:**

All configuration is done through the `.env` file. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `SV_TELEGRAM_BOT_TOKEN` | Yes | Your Telegram bot token from @BotFather |
| `SV_TELEGRAM_ALLOWED_USERS` | Yes | Comma-separated Telegram user IDs |
| `SV_DASHBOARD_PORT` | No | Dashboard port (default: 8080) |
| `SV_KLIPPER_HOST` | No | Klipper/Moonraker IP address |
| `SV_BAMBU_MQTT_HOST` | No | Bambu printer IP address |
| `SV_CAMERA_RTSP_URL` | No | RTSP URL for camera stream |

**Persistent data:**

Docker volumes are automatically created for:
- `sensing-voxel-data`: Agent state, calibration data
- `sensing-voxel-uploads`: Uploaded STL files
- `sensing-voxel-models`: ML models

**Updating:**

```bash
# Pull latest changes and rebuild
git pull
docker compose up -d --build
```

### Manual Installation

See `config/default.yaml` for configuration options:

- **Telegram**: Bot token and allowed user IDs
- **Klipper**: Moonraker host, port, and API key
- **Bambu**: MQTT and FTP credentials
- **Camera**: RTSP URL and calibration settings
- **Storage**: Data directory paths

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sensing_voxel --cov-report=html

# Run specific test file
pytest tests/unit/test_state_machine.py
```

### Project Structure

```
sensing-voxel/
├── src/
│   └── sensing_voxel/
│       ├── main.py                  # Entry point
│       ├── config.py                # Configuration
│       ├── core/                    # Core state machine, decision engine, persistence
│       ├── adapters/                # Printer adapters (Klipper, Bambu, base, command queue)
│       ├── bot/                     # Telegram bot (telegram, auth, decisions)
│       ├── api/                     # Dashboard REST/WS API
│       ├── llm/                     # LLM chat with tool use (50 tools)
│       ├── camera/                  # PTZ camera control (Tapo ONVIF)
│       ├── infra/                   # Shared utilities (retry, gcode validation)
│       ├── dashboard/               # Dashboard static files
│       ├── slicer/                  # Slicing pipeline (module-complete, unwired)
│       ├── vision/                  # Vision processing (module-complete, unwired)
│       ├── monitoring/              # Print monitoring (PrintMonitor wired), Z-offset, maintenance
│       ├── audio/                   # YAMNet anomaly detection (unwired, no config yet)
│       ├── calibration/             # Tower generation, visual scoring (unwired)
│       ├── storage/                 # SQLite job/calibration repositories (unwired)
│       ├── queue/                   # Multi-printer queue + schedulers (unwired)
│       └── modeling/                # Photo-to-print, image-to-3D, mesh repair (unwired)
├── tests/
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
├── config/
│   └── default.yaml                 # Default config
├── data/                            # Runtime data
└── docs/
    ├── PRD.md                       # Product requirements
    └── ARCHITECTURE.md              # Architecture doc
```

## Roadmap / Current Runtime Status

This section describes the currently wired runtime behavior, not just isolated
modules or tests.

- **Phase 0**: Stabilization
  - [x] Project scaffolding and install metadata
  - [x] Core data models, state machine, decision engine, and JSON persistence
  - [x] Security hardening for uploaded file paths and G-code command validation

- **Currently wired and usable**
  - [x] Klipper/Moonraker adapter: REST/WebSocket connection, telemetry, file upload,
        print lifecycle commands, temperatures, movement, macros, and basic system info
  - [x] Telegram allowlisted bot with `/start`, `/status`, `/reset_z_offset`,
        `/calibrate_camera`, `/ask`, `/clear_chat`, `/ctx`, `/audit`, `/tasks`,
        `/approvals`, and file upload handling
  - [x] Direct G-code upload workflow through Telegram:
        upload `.gcode`/`.g`/`.gco` → approval buttons → upload to printer → start print
  - [x] Web dashboard server with status, printer list, telemetry, alerts,
        WebSocket updates, chat endpoint, and camera stream route
  - [x] Tapo PTZ / RTSP camera support when configured
  - [x] Basic LLM chat when configured
  - [x] Background print monitoring (`PrintMonitor`): periodic camera snapshot + LLM vision
        anomaly detection, auto-pause on unacknowledged anomaly, auto-activates on PRINTING state
  - [x] Printer memo system: SVA stores physical attributes (bed surface, filament brand)
        in SQLite; rejects values that belong in printer.cfg (single source of truth)
  - [x] MAC address hardware fingerprinting: auto-discovers printer identity from Moonraker
  - [x] Web search (`web_search` tool via SearXNG): LLM performs autonomous multi-turn queries
  - [x] Tool cooldown: physical tools rate-limited to prevent hardware congestion
  - [x] Independent vision analysis: camera_snapshot performs bias-free LLM analysis
        in a separate context to prevent confirmation bias from chat history

- **Implemented as modules, but not fully wired into the main runtime**
  - [ ] STL/STEP/3MF slicing workflow: `SlicerPipeline` exists, but Telegram model
        uploads currently stop after staging the file and entering `RECEIVING`; the
        runtime does not yet call the slicer and advance to approval automatically
  - [ ] Monitoring orchestrator: external print detection,
        first-layer correction, audio anomaly handling, and filament monitoring are
        implemented in modules/tests but are not instantiated or started by `main.py`.
        Note: `PrintMonitor` (background camera check) IS wired; `MonitoringOrchestrator`
        (first-layer/sfilament/audio) is NOT.
  - [ ] Dashboard decision side effects: `/api/v1/decision` records a decision but
        does not yet perform the Telegram-equivalent upload/start/resume/cancel actions
  - [ ] Job history and queue dashboard APIs: route handlers exist, but repository and
        queue managers are not wired into app state by the main runtime
  - [ ] LLM tool execution on OpenAI-compatible backends: basic chat works, but the
        tool-use loop is not fully compatible with OpenAI-style tool call responses
  - [ ] Bambu runtime support: adapter code exists, but configuration and startup wiring
        are incomplete for production use
  - [ ] Photo-to-print / image-to-3D pipeline: modeling modules exist but are not exposed
        through Telegram or dashboard runtime workflows

- **Planned integration work**
  - Wire `SlicerPipeline.process_model()` into Telegram and dashboard upload flows
  - Start `MonitoringOrchestrator` from `main.py` with configured camera/detectors
  - Add first-layer test generation and approval/start workflow
  - Complete dashboard decision side effects
  - Wire SQLite job history and multi-printer queue manager
  - Finish OpenAI-compatible LLM tool-call protocol support upstream or via a minimal adapter

## License

GNU General Public License v3.0

This project is licensed under the GPL-3.0 license, which ensures that all derivatives and integrations remain open source. See the [LICENSE](LICENSE) file for details.

## Credits

Inspired by patterns from [OpenClaw](https://github.com/openclaw/openclaw).
