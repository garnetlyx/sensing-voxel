# Sensing Voxel - Agent Instructions

## Project Overview

AI-powered 3D printer monitoring and control system with real-time vision-based quality detection.

- **Runtime**: Python 3.12 + asyncio
- **Adapters**: Klipper (Moonraker REST+WS), Bambu (MQTT+FTP)
- **Interfaces**: Telegram bot, Web dashboard, LLM chat (Anthropic tool use)
- **Vision**: First-layer auto-correction, camera calibration, filament/tangle monitoring, USB camera
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
  vision/       - First-layer, calibration, filament, usb_camera (module-complete, unwired)
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

## Live Runtime Environment (verified 2026-07-10)

The system runs live against real hardware. All config is in `config/local.yaml` (gitignored).

### Services

| Service | Address | Status |
|---------|---------|--------|
| neptune4max (Klipper) | `printer.example:80` | ✅ Connected, WS subscribed |
| rpi-printer (Klipper) | `printer.example:80` | ✅ Connected, WS subscribed |
| Tapo C220 camera (ONVIF) | `192.168.68.67:2020` | ✅ Connected, PTZ working |
| Tapo C220 RTSP stream | `rtsp://threedp:garage10@192.168.68.67:554/stream1` | ✅ 1080p capture verified |
| LLM proxy (Anthropic protocol) | `http://host.example:4000` | ✅ MiniMax-M3 model |
| Telegram bot | `@example_bot` | ✅ Polling (ensure only one instance runs — duplicate `getUpdates` causes 409 Conflict) |
| Dashboard API | `http://localhost:8080` | ✅ 24 endpoints live |

### Start the server

```bash
source .venv/bin/activate
python -m sensing_voxel.main 2>&1 | tee /tmp/sv-server.log
```

Or in tmux: `tmux new-session -d -s sv-server -c <repo-root>` then send the above command.

### Verified working (as of 2026-07-10)

- LLM tool-use loop: `get_print_progress` returns real temps, `list_files` returns real G-code list from Moonraker
- Dual-printer Klipper adapter (REST + WS telemetry)
- Tapo camera RTSP capture (1920×1080 JPEG) + ONVIF PTZ step-move
- Dashboard REST API: health, printers, status, telemetry, chat, camera/move
- Tool audit log: `data/tool_audit.jsonl` records tool name + user ID + result excerpt
- State machine starts at IDLE, persistence loads from `data/agent_state.json`

### Known runtime gaps (confirmed by live test)

- `GET /api/v1/jobs/history` → returns empty (JobRepository not wired into app_state)
- `GET /api/v1/queue` → returns empty (QueueManager not wired)
- `POST /api/v1/decision` → records only, no upload/start/resume/cancel side effects
- Telegram STL/STEP upload → enters RECEIVING state, slicer never called
- LLM config has no `provider` field in local.yaml → defaults to `LlmProvider.MINIMAX` (Anthropic SDK path — this is the working path)

### LLM model multimodal verification

The model may be swapped (MiniMax-M3 today, Doubao/GPT/Ollama tomorrow). **Before relying on any vision feature (camera_snapshot tool, first-layer analysis), independently verify the connected model's multimodal capability** — do not assume it from model name alone.

Quick test (run from repo root with proxy reachable):

```bash
# Download a known test image
curl -sL -o /tmp/test-img.jpg "https://picsum.photos/id/237/400/300"

# Send image + question via the same Anthropic protocol SV uses
IMG_B64=$(base64 -i /tmp/test-img.jpg)
curl -s -m 60 "$LLM_BASE_URL/v1/messages" \
  -H "x-api-key: $LLM_API_KEY" \
  -H "content-type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d "{\"model\":\"$LLM_MODEL\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/jpeg\",\"data\":\"$IMG_B64\"}},{\"type\":\"text\",\"text\":\"What animal is this? Describe in detail.\"}]}]}"
```

**Pass criteria**: model correctly identifies the subject (a black Labrador puppy on wooden floor) without hallucinating objects that aren't there.

**If the model fails or returns text-only / errors**: vision features (camera_snapshot, first-layer visual analysis) are non-functional. Either switch to a multimodal-capable model or disable vision-dependent tools.

**Verified models**:
- `minimax-m3` via `http://host.example:4000` — ✅ verified 2026-07-10 (Labrador test + Tapo camera snapshot analysis)

### Verification plan

See `.project/module-verification/` for the full per-component verification checklist and go/no-go decisions on orphaned modules.

## LLM Integration

- 31+ tools covering Fluidd-equivalent functionality plus `run_shell`, `web_search` (SearXNG), background tasks, approvals
- Default backend: `volcengine-anthropic-doubao-seed-20-pro` via Anthropic-compatible proxy (MiniMax supported). OpenAI-compatible backends (Ollama) chat but the tool-use loop has two bugs (`llm/client.py` drops `tool_use` block `id` and doesn't set `stop_reason: "tool_use"`), so tool execution is non-functional there.
- System prompt includes Klipper command reference and live printer context
- High-risk shell/printer actions require approval-token replay
- Tool calls audit-logged to `data/tool_audit.jsonl`
- "Agent State" (orchestration FSM) != printer state — keep labels clear

## Known Issues

See `.project/EVAL.md` for the full review and `.project/TODO.md` for the actionable list. Highlights:

- **Test suite is green**: 1,181 passed / 0 failed / 18 skipped / 13 xfailed on default `pytest` run (resolved 2026-07-09, P0 in `.project/TODO.md`). Historical note: was 537/1,234 failing before P0 — root causes were `MockPrinterAdapter` missing `get_current_position`, `pytest-aiohttp` `TestClient` event-loop runner conflict, Playwright browser pin drift, stale `.venv` pip shebang.
- **Phase 2/3 modules are runtime-unwired**: `SlicerPipeline`, `MonitoringOrchestrator`, `AudioAnomalyDetector`, `FilamentMonitor`, `ImageTo3D` pipeline, `QueueManager`, SQLite repositories, and calibration automation all exist as tested code but `main.py` does not start them. Wiring plan in `.project/TODO.md` P1.
- **`audio/` has no `AudioConfig` dataclass** — blocks wiring.
- **`llm/tools.py` is 2,007 LOC** — split into a package before adding more tools.
- **Emergency cancel** now uses a dedicated session (resolved). Historical note: it once shared the primary session.
- Historical test-failure files (`test_adversarial_*`, `test_concurrency_bugs`, `test_decision_bugs`, `test_qa_critical_bugs`) were updated to guard the fixed behavior as regression tests during the P0 fix. See `.project/SESSION_LOG.md` for details.

## Documentation

Git-tracked:

- `README.md` — Onboarding, usage, and the "Roadmap / Current Runtime Status" section (source of truth for what is wired)
- `docs/PRD.md` — Product requirements
- `docs/ARCHITECTURE.md` — Technical architecture

Local-only (gitignored, managed by `/project`):

- `.project/TODO.md` — Current unfinished work only
- `.project/SESSION_LOG.md` — Checkpoints, decisions, work history
- `.project/EVAL.md` — Codebase review and evaluation
