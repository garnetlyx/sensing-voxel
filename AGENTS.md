# Sensing Voxel - Agent Instructions

## Terminology

- **SVA** (Sensing-Voxel Agent): the AI agent runtime that drives this system — the LLM tool-use loop, state machine, and all wired subsystems. When a user says "ask SVA to X", it means interact with the agent via dashboard chat or Telegram.

## Project Overview

AI-powered 3D printer monitoring and control system with real-time vision-based quality detection.

- **Runtime**: Python 3.12 + asyncio
- **Adapters**: Klipper (Moonraker REST+WS), Bambu (MQTT+FTP)
- **Interfaces**: Telegram bot, Web dashboard, LLM chat (Anthropic tool use)
- **Monitoring**: Camera snapshots assessed by the configured multimodal LLM through `PrintMonitor`
- **Camera**: Tapo C220 with ONVIF PTZ control

## Architecture

```
src/sensing_voxel/
  core/         - State machine, decision engine, persistence, data models
  safety/       - Validation, preview, authorization, and execution barriers
  adapters/     - Printer adapters (klipper.py, bambu.py, base.py)
  bot/          - Telegram bot interface (telegram.py)
  api/          - Web dashboard REST/WebSocket API
  llm/          - LLM chat with tool use (50 tools incl. web_search, print monitor, printer memo)
  camera/       - PTZ camera control (Tapo ONVIF)
  slicer/       - STL/STEP/3MF slicing pipeline + geometry analysis
  monitoring/   - PrintMonitor (background camera check + LLM anomaly detection)
  storage/      - SQLite job/memo repositories + Database
  dashboard/    - Static web dashboard files
  infra/        - Shared utilities (retry.py, gcode.py)
```

> **Historical cleanup (2026-07-13):** Superseded experimental packages and the old monitoring orchestrator were removed to enforce deduplication. Current camera analysis belongs to the multimodal LLM and `PrintMonitor`; consult git history for removed implementations.

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
- **Printer config safety**: **ALWAYS backup printer.cfg before any modification.** Moonraker auto-creates timestamped backups (e.g. `printer-20260516_075252.cfg`) but never overwrite printer.cfg without confirming a backup exists. SVA's `update_printer_config` tool MUST create a backup copy before writing. A single-line overwrite can brick the printer's startup.
- **G-code safety barriers** (MANDATORY, enforced in code):
  1. **G28 validation**: `start_print_with_monitor` refuses to print if G-code has no G28 (homing). This is a hard code barrier — not configurable.
  2. **Heating validation**: refuses to print if no M104/M109 (nozzle heating) in G-code.
  3. **Slice preview**: after slicing, SVA should show the user a summary (print time, filament, layer count, preview image if possible) before starting print.
  4. **Pre-print confirmation**: SVA must display material, build plate surface, hotend size, and z-offset, and get explicit user approval before starting.
  5. **First-layer monitoring**: during first layer, monitor checks every 1 minute (not 30). After first layer completes, switch to normal interval.
  6. **Z-offset sanity check**: if probe z_offset is 0.0 or homing_origin Z is positive, BLOCK print and warn user — this indicates broken/uninitialized probe state.
- **Deduplicate + consolidate**: No two modules may co-exist if they serve overlapping functions, even under different names or abstractions. When a replacement becomes canonical, delete its predecessor and migrate any still-needed logic. Renaming is not enough: shared responsibilities such as frame capture must have one owner. Every module in the codebase must have a unique, non-overlapping responsibility.
- **Local lifecycle state**: `.project/` holds `TODO.md`, `SESSION_LOG.md`, `EVAL.md` (gitignored)

## Local Runtime Configuration

Deployment-specific settings—including printer and camera endpoints, credentials, user or bot identifiers, model gateways, and service URLs—belong only in the ignored `config/local.yaml` or environment variables. Never copy values from local configuration, runtime logs, hardware discovery, or live service responses into tracked documentation, tests, examples, or defaults.

Use neutral placeholders for documentation and synthetic values for tests. Use RFC 5737 addresses (`192.0.2.0/24`, `198.51.100.0/24`, or `203.0.113.0/24`) whenever an example requires an IPv4 address.

Moonraker REST wraps results under `result.status.{object}`; unwrap with `data.get("status", data)`.

Before relying on model vision features, verify the configured model's multimodal capability in the local deployment. Do not record the endpoint, credentials, model deployment details, test images, or live verification results in tracked files.

Before any physical printer action, follow the project's safety barriers and verify current state through the configured adapter. Live hardware, camera, chat, and log checks are deployment operations and must use local configuration rather than commands or addresses embedded in repository documentation.

## LLM Integration

- 50 tools covering Fluidd-equivalent functionality plus `run_shell`, `web_search` (SearXNG), background tasks, approvals, printer memo, print monitoring, hardware fingerprinting
- Default backend: `volcengine-anthropic-doubao-seed-20-pro` via Anthropic-compatible proxy (MiniMax supported). OpenAI-compatible backends (Ollama) chat but the tool-use loop has two bugs (`llm/client.py` drops `tool_use` block `id` and doesn't set `stop_reason: "tool_use"`), so tool execution is non-functional there.
- System prompt includes Klipper command reference and live printer context
- High-risk shell/printer actions require approval-token replay
- Tool calls audit-logged to `data/tool_audit.jsonl`
- "Agent State" (orchestration FSM) != printer state — keep labels clear

## Known Issues

See `README.md` for current runtime status and `.project/TODO.md` for the actionable local backlog. Highlights:

- **Reported default test baseline**: 719 passed / 18 skipped / 3 xfailed. This count was supplied for the documentation refresh; no persisted local pytest result artifact independently confirms it. Rerun `pytest` for current evidence.
- **Current monitoring path**: `PrintMonitor` performs background camera checks with multimodal LLM anomaly assessment; deleted first-layer correction and legacy monitoring modules are historical only.
- **Remaining wiring work**: Telegram model-upload handling exists but is not composed into the runtime. `SlicerPipeline` already detects available PrusaSlicer or OrcaSlicer CLIs. Runtime storage, including `JobRepository`, is already composed.
- **`llm/tools.py` is large** — split it only as a dedicated, behavior-preserving refactor before adding substantial tool surface.
- **Emergency cancel** uses a dedicated session; references to shared-session behavior are historical.

## Documentation

Git-tracked:

- `README.md` — Onboarding, usage, and the "Current Limitations" section (source of truth for what is wired)
- `docs/PRD.md` — Product requirements
- `docs/ARCHITECTURE.md` — Technical architecture

Local-only (gitignored, managed by `/project`):

- `.project/TODO.md` — Current unfinished work only
- `.project/SESSION_LOG.md` — Checkpoints, decisions, work history
- `.project/EVAL.md` — Codebase review and evaluation
