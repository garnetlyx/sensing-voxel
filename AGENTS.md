# Sensing Voxel - Agent Instructions

## Terminology

- **SVA** (Sensing-Voxel Agent): the AI agent runtime that drives this system — the LLM tool-use loop, state machine, and all wired subsystems. When a user says "ask SVA to X", it means interact with the agent via dashboard chat or Telegram.

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
  llm/          - LLM chat with tool use (50 tools incl. web_search, print monitor, printer memo)
  camera/       - PTZ camera control (Tapo ONVIF)
  slicer/       - STL/STEP/3MF slicing pipeline + geometry analysis
  monitoring/   - PrintMonitor (background camera check + LLM anomaly detection)
  storage/      - SQLite job/memo repositories + Database
  dashboard/    - Static web dashboard files
  infra/        - Shared utilities (retry.py, gcode.py)
```

> **Deleted modules** (2026-07-13): `vision/`, `audio/`, `calibration/`, `queue/`, `modeling/`, `monitoring/monitor.py+z_offset.py+maintenance.py` — all removed to enforce deduplication. Vision analysis is now done via LLM multimodal (camera_snapshot tool + PrintMonitor). These modules exist in git history if needed.

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
- **Deduplicate + consolidate**: No two modules may co-exist if they serve overlapping functions — even under different names or abstractions. If a new module replaces an old one (e.g. PrintMonitor replaces MonitoringOrchestrator, LLM multimodal replaces vision/), **delete the old one immediately and migrate any still-needed logic into the new module**. Renaming is not enough — if `vision/camera.py` (RTSP stream) and `monitoring/print_monitor.py` (RTSP snapshot) both capture camera frames, consolidate into one place. Every module in the codebase must have a unique, non-overlapping responsibility.
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
| SearXNG (web search) | `http://localhost:8888` (Docker) | ✅ JSON API enabled, web_search tool wired |

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
- `web_search` tool: SearXNG instance at `localhost:8888` (Docker), LLM performs multi-turn progressive queries autonomously
- `camera_snapshot` tool: RTSP → base64 → MiniMax-M3 multimodal analysis (cross-verified against independent look_at)
- Tool cooldown: physical tools rate-limited (default 1.0s between identical tool calls)
- Print monitoring (`PrintMonitor`): auto-activates when printer enters PRINTING state, periodic camera snapshot + LLM anomaly detection, auto-pause on unacknowledged anomaly (configurable timeout)
- Printer memo system: SQLite key-value storage for physical attributes not in Klipper config (bed surface, filament brand); rejects config keys to maintain single source of truth
- MAC address fingerprinting: auto-discovers printer hardware identity from Moonraker `machine/system_info`, stores in memo DB for persistent identification
- Independent vision analysis: `camera_snapshot` performs a bias-free LLM analysis in a separate context (no chat history) to prevent confirmation bias
- `update_printer_config` tool: SVA can modify printer.cfg with mandatory backup + approval flow
- `approve_pending` tool: helps LLM replay approval tokens reliably

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

### Live debugging workflow

When a bug is found or behavior needs verification, follow this loop:

```
┌─► 1. DISCOVER: identify the problem (log error, wrong behavior, user report)
│
│   2. INVESTIGATE: gather evidence in parallel
│      • Server log:   tail -50 /tmp/sv-server.log
│      • Camera frame: capture via RTSP → /tmp/sv-camera-snapshot.jpg → look_at
│      • Hardware:     curl Moonraker REST endpoints, ping devices
│      • Code:         grep / read the relevant module
│      • LLM state:    GET /api/v1/chat/context, tool_audit.jsonl
│
│   3. FIX: make the minimal code change (never refactor while fixing)
│      • Run unit tests: pytest tests/unit/test_<module>.py -x -q
│
│   4. RESTART: reload the server with the fix
│      • Ctrl-C in tmux, re-run python -m sensing_voxel.main | tee /tmp/sv-server.log
│      • Verify startup log: adapters connected, camera ONVIF, LLM initialized
│
│   5. REPRODUCE AS USER: act as a human user via Playwright dashboard
│      • Navigate to http://localhost:8080
│      • Type the triggering message in the chat input
│      • Observe: LLM response, tool calls in log, camera movement, state changes
│      • For vision: capture RTSP frame → look_at → cross-check LLM's description
│
│   6. VERIFY OUTCOME
│      ├─ FIXED → 7. DOCUMENT + COMMIT
│      │         • Update AGENTS.md / .project/module-verification/ if behavior changed
│      │         • git add + commit with conventional-commit message
│      │         • Confirm: pytest still green, server still running
│      │
│      └─ STILL BROKEN or NEW BUG → back to step 1 (↑)
└──────────────────────────────────────────────────────────────────────────
```

Key tools for each step:
- **Investigate**: `tail /tmp/sv-server.log`, `curl localhost:8080/api/v1/*`, RTSP capture + `look_at`
- **Reproduce**: Playwright browser at `http://localhost:8080` — type in `#chat-input`, click `#chat-send`
- **Verify**: `grep "Tool call:\|cooldown\|error" /tmp/sv-server.log`, `pytest tests/unit/ -x -q`

### Pre-action state check (MANDATORY)

**Before SVA or the operator performs ANY action on a printer** (start print, pause, cancel, set temp, move axis, run gcode, modify config), the current state MUST be read first. This is not optional — it is a hard protocol hook.

What to read:
1. **Printer state**: `curl -s http://<printer>:80/printer/objects/query?print_stats&extruder&heater_bed`
2. **Camera snapshot**: capture RTSP frame → look_at (if camera available)
3. **Server log tail**: `tail -20 /tmp/sv-server.log`

Why:
- Prevents acting on stale assumptions (printer may have finished/failed since last check)
- Catches unexpected states (someone else started a print, print failed, etc.)
- Provides evidence trail for post-mortem analysis
- The "pre-print validation" feature depends on this — SVA should never start a print without first reading printer.cfg, memo DB, and current printer state

Implementation: SVA's system prompt instructs the LLM to always call `get_print_progress` before any printer action. The operator (you, the AI agent reading this file) must do the equivalent via direct API calls before driving SVA to act.

## LLM Integration

- 50 tools covering Fluidd-equivalent functionality plus `run_shell`, `web_search` (SearXNG), background tasks, approvals, printer memo, print monitoring, hardware fingerprinting
- Default backend: `volcengine-anthropic-doubao-seed-20-pro` via Anthropic-compatible proxy (MiniMax supported). OpenAI-compatible backends (Ollama) chat but the tool-use loop has two bugs (`llm/client.py` drops `tool_use` block `id` and doesn't set `stop_reason: "tool_use"`), so tool execution is non-functional there.
- System prompt includes Klipper command reference and live printer context
- High-risk shell/printer actions require approval-token replay
- Tool calls audit-logged to `data/tool_audit.jsonl`
- "Agent State" (orchestration FSM) != printer state — keep labels clear

## Known Issues

See `.project/EVAL.md` for the full review and `.project/TODO.md` for the actionable list. Highlights:

- **Test suite is green**: 639 passed / 0 failed / 13 xfailed on default `pytest` run. Reduced from 1,181 after deleting orphaned module tests (vision, audio, calibration, queue, modeling, old monitoring).
- **Orphaned modules DELETED** (2026-07-13): vision/, audio/, calibration/, queue/, modeling/, monitoring/monitor.py+z_offset.py+maintenance.py — all removed. Rationale: either superseded by PrintMonitor (vision → LLM multimodal), blocked by missing hardware/config (audio, calibration), not needed for single-user (queue), or too many bugs + external deps (modeling). Git history preserved.
- **Remaining wiring work**: `SlicerPipeline` needs PrusaSlicer CLI install + Telegram upload flow integration. `JobRepository` needs dashboard app_state wiring.
- **`llm/tools.py` is 2,630 LOC** — split into a package before adding more tools.
- **Emergency cancel** now uses a dedicated session (resolved). Historical note: it once shared the primary session.

## Documentation

Git-tracked:

- `README.md` — Onboarding, usage, and the "Roadmap / Current Runtime Status" section (source of truth for what is wired)
- `docs/PRD.md` — Product requirements
- `docs/ARCHITECTURE.md` — Technical architecture

Local-only (gitignored, managed by `/project`):

- `.project/TODO.md` — Current unfinished work only
- `.project/SESSION_LOG.md` — Checkpoints, decisions, work history
- `.project/EVAL.md` — Codebase review and evaluation
