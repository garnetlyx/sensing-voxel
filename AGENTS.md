# Sensing Voxel - Agent Instructions

## Project Overview

AI-powered 3D printer monitoring and control system with real-time vision-based quality detection.

- **Runtime**: Python 3.12 + asyncio
- **Adapters**: Klipper (Moonraker REST+WS), Bambu (MQTT+FTP)
- **Interfaces**: Telegram bot, Web dashboard, LLM chat (Anthropic tool use)
- **Vision**: NOTHING
- **Camera**: Tapo C220 with ONVIF PTZ control

## Architecture

```
src/sensing_voxel/
  core/         - State machine, decision engine, persistence, data models
  adapters/     - Printer adapters (klipper.py, bambu.py, base.py, command_queue.py)
  bot/          - Telegram bot (telegram.py, auth.py, decisions.py)
  api/          - Web dashboard REST/WebSocket API
  llm/          - LLM chat with tool use (31+ tools incl. web_search)
  camera/       - PTZ camera control (Tapo ONVIF)
  slicer/       - STL/STEP/3MF slicing pipeline + geometry analysis (module-complete, unwired)
  vision/       - NOTHING, first-layer, calibration, filament, usb_camera (module-complete, unwired)
  monitoring/   - Print monitoring orchestrator, Z-offset, predictive maintenance (module-complete, unwired)
  audio/        - YAMNet anomaly detection (Phase 2, module-complete, unwired, no AudioConfig yet)
  calibration/  - Tower generation, visual scoring, config injection (Phase 2, module-complete, unwired)
  storage/      - SQLite job/calibration repositories (Phase 2, module-complete, unwired)
  queue/        - Multi-printer queue + schedulers (Phase 2, module-complete, unwired)
  modeling/     - Photo-to-print pipeline, image-to-3D, mesh repair (Phase 3, module-complete, unwired)
  dashboard/    - Static web dashboard files
  infra/        - Shared utilities (retry.py, gcode.py)
```

> **Wired vs. module-complete.** Only `core`, `adapters`, `bot`, `api`, `llm`, `camera`, and `infra` are instantiated by `main.py` today. Every other module is real, tested code that is not yet started at runtime. See `docs/ARCHITECTURE.md` §9 and `.project/TODO.md`.

## Development

```bash
# Setup (recreate venv if pip shebang is stale)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium   # optional: for dashboard E2E tests

# Run
python -m sensing_voxel.main

# Test
pytest
pytest --cov=sensing_voxel --cov-report=html
```

## Key Conventions

- **Config**: YAML-based (`config/default.yaml`, override with `config/local.yaml`); Docker path uses `SV_*` env vars via `docker-compose.yml`
- **State persistence**: JSON atomic snapshots (`data/agent_state.json`) with `schema_version`
- **Adapter pattern**: All printers implement `PrinterAdapter` ABC in `adapters/base.py` (includes `get_current_position`)
- **Decision safety**: Lock-guarded CAS via `consume_decision()` in `core/decision.py`
- **Emergency path**: Dedicated session for emergency cancel, bypasses the command queue
- **G-code validation**: Shared `validate_gcode_command()` in `infra/gcode.py` (blocks M112/M410/M80/M81/M23, validates axis + temperature bounds)
- **Local lifecycle state**: `.project/` holds `TODO.md`, `SESSION_LOG.md`, `EVAL.md` (gitignored)

## Printer Connectivity

- Neptune 4 Max (Klipper): `printer.example:80` — no auth required
- Second printer: `printer.example:80`
- Moonraker REST wraps results under `result.status.{object}` — unwrap with `data.get("status", data)`

## LLM Integration

- 31+ tools covering Fluidd-equivalent functionality plus `run_shell`, `web_search` (SearXNG), background tasks, approvals
- Default backend: `volcengine-anthropic-doubao-seed-20-pro` via Anthropic-compatible proxy (MiniMax and Ollama also supported)
- System prompt includes Klipper command reference and live printer context
- High-risk shell/printer actions require approval-token replay
- Tool calls audit-logged to `data/tool_audit.jsonl`
- "Agent State" (orchestration FSM) != printer state — keep labels clear

## Known Issues

See `.project/EVAL.md` for the full review and `.project/TODO.md` for the actionable list. Highlights:

- **Test suite is broken**: 537 of 1,234 tests currently fail. Root causes: `MockPrinterAdapter` missing newly-added abstract methods (`get_current_position`), `pytest-aiohttp` `TestClient` event-loop runner conflict, Playwright browser pin drift, stale `.venv` pip shebang. Fixes are mechanical (`.project/TODO.md` P0).
- **Phase 2/3 modules are runtime-unwired**: `SlicerPipeline`, `MonitoringOrchestrator`, `AudioAnomalyDetector`, `FilamentMonitor`, `ImageTo3D` pipeline, `QueueManager`, SQLite repositories, and calibration automation all exist as tested code but `main.py` does not start them. Wiring plan in `.project/TODO.md` P1.
- **`monitoring/monitor.py` imports vision at module level** — must be moved to lazy init before the orchestrator is wired.
- **`audio/` has no `AudioConfig` dataclass** — blocks wiring.
- **`llm/tools.py` is 2,007 LOC** — split into a package before adding more tools.
- **Emergency cancel** now uses a dedicated session (resolved). Historical note: it once shared the primary session.
- Historical test-failure files (`test_adversarial_*`, `test_concurrency_bugs`, `test_decision_bugs`, `test_qa_critical_bugs`) are no longer the only failing tests — the regression is broader now. Consult `.project/EVAL.md` for the current matrix.

## Documentation

Git-tracked:

- `README.md` — Onboarding, usage, and the "Roadmap / Current Runtime Status" section (source of truth for what is wired)
- `docs/PRD.md` — Product requirements
- `docs/ARCHITECTURE.md` — Technical architecture

Local-only (gitignored, managed by `/project`):

- `.project/TODO.md` — Current unfinished work only
- `.project/SESSION_LOG.md` — Checkpoints, decisions, work history
- `.project/EVAL.md` — Codebase review and evaluation
