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
  llm/          - LLM chat with tool use (26 tools)
  camera/       - PTZ camera control (Tapo ONVIF)
  vision/       - NOTHING detection, first-layer classifier, calibration
  slicer/       - STL slicing pipeline
  monitoring/   - Print monitoring orchestrator, Z-offset control
  dashboard/    - Static web dashboard files
  infra/        - Shared utilities (retry.py)
```

## Development

```bash
# Setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run
python -m sensing_voxel.main

# Test
pytest
pytest --cov=sensing_voxel --cov-report=html
```

## Key Conventions

- **Config**: YAML-based (`config/default.yaml`, override with `config/local.yaml`)
- **State persistence**: JSON atomic snapshots (`data/agent_state.json`)
- **Adapter pattern**: All printers implement `PrinterAdapter` ABC in `adapters/base.py`
- **Decision safety**: Lock-guarded CAS via `consume_decision()` in `core/decision.py`
- **Emergency path**: Dedicated session for emergency cancel, bypasses command queue

## Printer Connectivity

- Neptune 4 Max (Klipper): `printer.example:80` — no auth required
- Second printer: `printer.example:80`
- Moonraker REST wraps results under `result.status.{object}` — unwrap with `data.get("status", data)`

## LLM Integration

- 26 tools covering Fluidd-equivalent functionality
- MiniMax-M2.5 via `api.minimaxi.com/anthropic` proxy
- System prompt includes Klipper command reference
- "Agent State" (orchestration FSM) != printer state — keep labels clear

## Known Issues

- Pre-existing test failures in: test_adversarial_edge_cases, test_adversarial_concurrency, test_concurrency_bugs, test_decision_bugs, test_qa_critical_bugs
- Emergency cancel shares primary session (should use dedicated session)
- `telegram.py` has `hasattr` guards instead of proper DI
- Vision/slicer/monitoring modules are stubs (Phase 1 incomplete)

## Documentation

- `docs/PRD.md` — Product requirements
- `docs/ARCHITECTURE.md` — Technical architecture
- `docs/TASKS.md` — Task breakdown by phase
- `docs/EVAL.md` — Codebase review report
- `docs/SESSION_LOG.md` — Session checkpoints
