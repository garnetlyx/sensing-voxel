# Sensing Voxel

AI-powered 3D printer monitoring and control system with real-time vision-based quality detection.

## Features

- 🤖 **Telegram Bot Interface**: Upload STL files, receive notifications, and control prints
- 🍝 **NOTHING Detection**: NOTHING
- 🔧 **First-Layer Auto-Correction**: Z-offset adjustment for perfect first layers
- 📸 **Camera Calibration**: Checkerboard-based bed-level calibration
- 🖨️ **Multi-Printer Support**: Klipper/Moonraker and Bambu printers
- 🛡️ **External Print Detection**: Monitor prints started outside the agent
- 📊 **Web Dashboard**: Real-time monitoring UI with WebSocket updates

## Architecture

```
Telegram Bot <-> Agent Core (state machine + policy + decision engine)
                      |
      +---------------+-------------------+
      |               |                   |
 Slicer Pipeline   Vision Monitor     Printer Adapter (abstract)
 (PrusaSlicer CLI) (RTSP → OpenCV)     ├── KlipperAdapter (Moonraker REST/WS)
                   ├── NOTHING       └── BambuAdapter (Local MQTT + FTP)
                   │   (NOTHING
                   ├── FirstLayer
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

### Docker Deployment

```bash
docker-compose up -d
```

## Configuration

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
│       ├── core/                    # Core state machine
│       ├── adapters/                # Printer adapters
│       ├── bot/                     # Telegram bot
│       ├── api/                     # Dashboard REST/WS API
│       ├── dashboard/               # Dashboard static files
│       ├── slicer/                  # Slicing pipeline
│       ├── vision/                  # Vision processing
│       └── monitoring/              # Print monitoring
├── tests/
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
├── config/
│   └── default.yaml                 # Default config
├── data/                            # Runtime data
└── docs/
    ├── PRD.md                       # Product requirements
    ├── ARCHITECTURE.md              # Architecture doc
    └── TASKS.md                     # Development tasks
```

## Roadmap

- **Phase 0**: Stabilization (environment, compilation errors, critical bugs)
  - [x] Fix development environment (venv + pip install)
  - [x] Fix compilation errors (bambu.py, klipper.py, command_queue.py)
  - [x] Bug triage & deduplication
  - [x] Fix CRITICAL security bugs (G-code injection, path traversal, job ID collision)
  - [x] Fix HIGH severity bugs

- **Phase 1 (MVP)**: Core agent-started print workflow
  - [x] Project scaffolding
  - [x] Core data models
  - [x] State machine
  - [x] Decision engine
  - [x] State persistence
  - [x] Printer adapter interface
  - [x] External print detection
  - [ ] Klipper adapter (~70%)
  - [ ] Bambu adapter (~80%, blocked by compilation error)
  - [ ] Telegram bot (~50%)
  - [ ] Slicer pipeline
  - [ ] Vision: NOTHING detection
  - [ ] Vision: camera calibration
  - [ ] Vision: first-layer classifier
  - [ ] Print monitoring orchestrator (~60%)
  - [ ] Integration tests

- **Phase 2**: Enhancements
  - Full Bambu adapter parity
  - Audio anomaly detection
  - STEP/3MF support
  - Closed-loop calibration
  - SQLite job history

- **Phase 3**: Polish
  - Multi-printer queue scheduling
  - [x] Dashboard UI
  - Predictive maintenance

## License

MIT License

## Credits

Inspired by patterns from [OpenClaw](https://github.com/openclaw/openclaw).
