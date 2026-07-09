# Sensing Voxel - Architecture Document

**Version**: 1.3
**Date**: 2026-03-27
**Source**: `.design/plan.md` v10 + iterative runtime hardening
**Last Review**: 2026-03-27 — LLM runtime hardening, tool audit, shared session context, shell access

---

## 1. Overview

A modular async architecture with a shared Agent Core state machine and printer-adapter abstraction. **MVP deploys one Agent Core instance per printer.** Each instance has its own state machine, Telegram command namespace (or printer-selection prefix), and `agent_state.json` file. Multi-printer coordination is deferred to Phase 2.

The architecture is designed in three phases:
- **Phase 1 (MVP)**: Print automation — Telegram intake, slicing, monitoring, printer control.
- **Phase 2**: Platform maturity — audio detection, calibration automation, filament monitoring, multi-printer.
- **Phase 3**: Photo-to-print — Image-to-3D generation, mesh repair, end-to-end pipeline.

```text
Phase 1 (MVP):

Telegram Bot <-> Agent Core (state machine + policy + decision engine)
                      |
      +---------------+-------------------+
      |               |                   |
 Slicer Pipeline   Vision Monitor     Printer Adapter (abstract)
 (PrusaSlicer/     (RTSP → OpenCV)     ├── KlipperAdapter (Moonraker REST/WS)
  OrcaSlicer CLI)  ├── NOTHING       └── BambuAdapter (Local MQTT + FTP)
                   │   (NOTHING
                   ├── FirstLayer
                   │   "God Hand"
                   │   (OpenCV + SET_GCODE_OFFSET Z_ADJUST)
                   └── Calibration
                       (Checkerboard)

Phase 2 additions:

                   ├── Audio Monitor (YAMNet → M112)
                   │   Stepper skip / glass crack / filament grind
                   ├── Filament Monitor (USB cam → spool volume + tangle detection)
                   └── Calibration Automation (tower scoring → SAVE_CONFIG)

Phase 3 additions:

 Photo/Sketch ──→ Image-to-3D API ──→ Mesh Repair ──→ [enters Phase 1 pipeline]
 (Telegram)       (TripoSR/Meshy/     (MeshLab/Blender
                   Tripo3D)            headless scripts)
```

---

## 2. Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Runtime | Python 3.12 + `asyncio` | Mature async + CV/ML ecosystem |
| Bot | `python-telegram-bot` | Robust async Telegram integration |
| Slicing | PrusaSlicer CLI (headless) | Headless automation and profile support |
| Vision | OpenCV + NOTHING
| Camera | RTSP via OpenCV `VideoCapture` | Network IP camera support |
| Camera PTZ | ONVIF via `python-onvif-zeep` | Tapo C220 pan/tilt/zoom control |
| Klipper Control | Moonraker REST + WebSocket | Standard Klipper control plane |
| Bambu Control | Local MQTT + FTP/FTPS | Native local integration path |
| LLM | Anthropic SDK (`anthropic`) | Tool-use agentic loop for natural language control |
| Storage (MVP) | JSON atomic snapshots | Simple state persistence, no DB dependency |
| Storage (Phase 2) | SQLite | Job history, calibration records, analytics |
| Config | YAML/TOML | Human-readable configuration |
| Deployment | Docker Compose / systemd | Reliable SBC deployment |
| Testing | pytest + pytest-asyncio | Async-native test framework |

---

## 3. Directory Structure

```
sensing-voxel/
├── src/
│   └── sensing_voxel/
│       ├── __init__.py
│       ├── main.py                  # Entry point, async event loop
│       ├── config.py                # Configuration loading (YAML/TOML)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── state_machine.py     # Agent state machine + transitions
│       │   ├── decision.py          # consume_decision() + lock management
│       │   ├── persistence.py       # JSON atomic state snapshots
│       │   └── models.py            # State enums, data classes, events
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── base.py              # PrinterAdapter abstract base class
│       │   ├── klipper.py           # KlipperAdapter (Moonraker REST/WS)
│       │   ├── bambu.py             # BambuAdapter (MQTT + FTP)
│       │   └── command_queue.py     # Priority command queue + emergency bypass
│       ├── bot/
│       │   ├── __init__.py
│       │   ├── telegram.py          # Telegram bot handlers
│       │   ├── auth.py              # User allowlist authentication
│       │   └── decisions.py         # Inline keyboard decision UI
│       ├── api/
│       │   ├── __init__.py
│       │   ├── server.py            # aiohttp web server setup
│       │   ├── routes.py            # REST API endpoints
│       │   ├── websocket.py         # WebSocket manager for dashboard
│       │   ├── camera_proxy.py      # RTSP→MJPEG proxy for browser
│       │   ├── auth.py              # API token authentication
│       │   └── serializers.py       # State/telemetry JSON serializers
│       ├── dashboard/
│       │   └── static/              # HTML/CSS/JS dashboard UI
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py            # Async Anthropic SDK wrapper
│       │   ├── chat.py              # ChatSession + agentic tool-use loop
│       │   ├── context.py           # PrinterContextCollector (system prompt)
│       │   ├── runtime.py           # Request context, audit log, compaction helpers
│       │   └── tools.py             # 31 tool definitions + ToolExecutor
│       ├── camera/
│       │   ├── __init__.py
│       │   └── tapo.py              # Tapo C220 ONVIF PTZ controller
│       ├── slicer/
│       │   ├── __init__.py
│       │   ├── pipeline.py          # STL → G-code pipeline
│       │   └── metadata.py          # FirstLayerParams/Bounds extraction
│       ├── vision/
│       │   ├── __init__.py
│       │   ├── camera.py            # RTSP stream capture
│       │   ├── calibration.py       # Checkerboard calibration lifecycle
│       │   ├── NOTHING.py         # NOTHING
│       │   └── first_layer.py       # First-layer classifier + Z-offset control
│       └── monitoring/
│           ├── __init__.py
│           ├── monitor.py           # Print monitoring orchestrator
│           └── z_offset.py          # Z-offset baseline/restore/reset
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/                         # Live system tests
│   └── conftest.py
├── config/
│   └── default.yaml                # Default configuration
├── data/                            # Runtime data (gitignored)
│   └── agent_state.json
├── models/                          # ML model files (gitignored)
│   └── NOTHING
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── TASKS.md
│   └── SESSION_LOG.md
├── .design/                         # Design artifacts (source of truth)
├── pyproject.toml
├── Dockerfile
└── docker-compose.yaml
```

---

## 4. Core Components

### 4.1 Agent Core State Machine

Primary states: `IDLE`, `RECEIVING`, `SLICING`, `AWAITING_APPROVAL`, `UPLOADING`, `PRINTING`, `MONITORING`, `PAUSED_PENDING_USER`, `COMPLETE`, `FAILED`, `EMERGENCY_FAILED`, `ERROR`.

#### State Transition Table

Any transition not listed is **invalid by default**. Invalid transition attempts must be logged and rejected.

```
IDLE -> RECEIVING              : Telegram file upload received
IDLE -> MONITORING             : External print detected (debounced)
IDLE -> PAUSED_PENDING_USER    : External print detected already paused

RECEIVING -> SLICING           : File validated successfully
RECEIVING -> ERROR             : File validation failure (bad type/size/corrupt)

SLICING -> AWAITING_APPROVAL   : Slicing complete, summary ready
SLICING -> ERROR               : Slicer process failure or timeout

AWAITING_APPROVAL -> UPLOADING : User approves print
AWAITING_APPROVAL -> IDLE      : User rejects or timeout (10min default)

UPLOADING -> PRINTING          : G-code uploaded, print started
UPLOADING -> ERROR             : Upload or start_print command rejected

PRINTING -> MONITORING         : Print confirmed active, monitoring begins
PRINTING -> FAILED             : Printer error event before monitoring starts
PRINTING -> ERROR              : Command failure during print start

MONITORING -> PAUSED_PENDING_USER : Moderate NOTHING confidence (pause + ask)
MONITORING -> PAUSED_PENDING_USER : First-layer saturation exit (correction maxed)
MONITORING -> EMERGENCY_FAILED    : High NOTHING confidence (emergency cancel)
MONITORING -> COMPLETE            : Print completed successfully
MONITORING -> FAILED              : Printer error event during monitoring

PAUSED_PENDING_USER -> MONITORING       : User resumes print
PAUSED_PENDING_USER -> FAILED           : User cancels or timeout (5min default)
PAUSED_PENDING_USER -> EMERGENCY_FAILED : Emergency cancel while paused

COMPLETE -> IDLE               : Cleanup + baseline restore + snapshot
FAILED -> IDLE                 : Cleanup + notification + snapshot
EMERGENCY_FAILED -> IDLE       : Urgent notification + cleanup + snapshot

ERROR -> IDLE                  : Auto-cleanup after error notification (recoverable)
```

Error handling policy:
- `ERROR` is auto-recoverable: agent notifies user, cleans up resources, and transitions to `IDLE`.
- `FAILED` requires printer state confirmation before returning to `IDLE`.
- `EMERGENCY_FAILED` sends urgent notification and marks job for post-mortem before `IDLE`.

#### External Print Policy

Canonical rule: **External prints are NOTHING-only monitoring.** First-layer auto-correction is never enabled for external prints even if metadata can be parsed. This rule is asserted in integration tests.

### 4.2 Decision Idempotency

All decision paths use a single helper:

```python
async def consume_decision(
    job_id: str,
    decision_token: str,
    action: DecisionAction,
    expected_state: AgentState
) -> ConsumeResult:
```

**Mandatory sequence**:
1. Acquire `decision_lock` (`asyncio.Lock`, keyed per `job_id`).
2. Re-check guard: current state, current job, token match, `decision_consumed == false`.
3. If guard passes: set `decision_consumed=true`, `decision_action=<action>`, `decision_consumed_at=<timestamp>`.
4. Persist state snapshot atomically to disk.
5. Release lock.
6. Execute side effects (printer command, Telegram message edits, transitions).

**Rules**:
- Timeout handlers, inline callbacks, text commands, and recovery code must all call this helper.
- If guard fails at step 2, return `already_handled` and perform no side effects.
- Side effects never run while lock is held.

**Timeout configuration** (stored in config, per-decision-type):
- `approval_timeout_seconds`: 600 (10 minutes)
- `NOTHING`: 300 (5 minutes)

### 4.3 State Persistence

`data/agent_state.json` is written on every transition using **temp-file + atomic rename**.

**MVP**: JSON-only. SQLite deferred to Phase 2 for job history/calibration records/analytics.

**IDLE reset contract** — on transition to `IDLE`, write a full snapshot:

Fields **reset**:
- `job_id`, `job_file`, `expires_at`, `pending_action`, `decision_token`, `decision_consumed`, `decision_action`, `printing_nozzle_temp`, `safe_pause_active`, `current_z_offset`, `z_offset_baseline`.

Fields **preserved**:
- `z_offset_dirty`
- `last_known_z_offset_baseline`
- `moonraker_compat_mode`
- `last_error_summary`

### 4.4 Z-Offset Baseline/Restore

For agent-started jobs with first-layer correction enabled:
1. Capture baseline at monitoring start from normalized printer object query.
2. Track cumulative print-local offset.
3. On terminal transition, restore baseline and verify within tolerance (±0.005mm).
4. If verification fails, set `z_offset_dirty=true` and block new agent-started prints.

**Saturation exit**: If cumulative correction reaches clamp boundary (`±0.05mm`) and classifier still reports poor quality for **3 consecutive evaluations**, agent pauses print via `PAUSED_PENDING_USER` and notifies user with current offset and classification result.

External prints: no first-layer correction, no baseline capture/restore.

**`/reset_z_offset` command**:
- Signature: `/reset_z_offset [target_mm]`
- Default target: `last_known_z_offset_baseline` if present.
- If baseline unavailable and no argument: return guidance, do not modify printer state.
- Explicit zero requires `/reset_z_offset 0`.
- Dirty flag cleared only after set+verify success.

### 4.5 Printer Adapter Interface Contract

Abstract base class that both `KlipperAdapter` and `BambuAdapter` implement:

```python
class PrinterAdapter(ABC):
    # --- Commands ---
    async def start_print(self, gcode_path: str) -> CommandResult
    async def pause_print(self) -> CommandResult
    async def resume_print(self) -> CommandResult
    async def cancel_print(self) -> CommandResult
    async def emergency_cancel(self) -> CommandResult  # Bypasses queue
    async def upload_file(self, local_path: str, remote_name: str) -> CommandResult
    async def get_z_offset(self) -> float
    async def set_z_offset(self, offset_mm: float) -> CommandResult

    # --- Queries ---
    async def get_printer_state(self) -> NormalizedPrinterState
    async def get_print_progress(self) -> PrintProgress
    async def is_connected(self) -> bool

    # --- Lifecycle ---
    async def connect(self) -> None
    async def disconnect(self) -> None

    # --- Events (adapter emits these) ---
    # state_changed(old: NormalizedPrinterState, new: NormalizedPrinterState)
    # telemetry_update(data: TelemetryData)
    # connection_lost()
    # connection_restored()
```

**Error model**:
- `AdapterCommandError`: Command failed (e.g., printer rejected command). Includes `retryable: bool`.
- `AdapterConnectionError`: Transport-level failure. Triggers reconnection logic.
- Emergency cancel uses a dedicated client/session, never shares the primary connection.

**Normalized printer states** (all adapter implementations must map to these):

| Source | Raw Values | Normalized Enum |
|--------|------------|-----------------|
| Moonraker `print_stats.state` | `printing` | `PRINTING` |
| Moonraker `print_stats.state` | `paused` | `PAUSED` |
| Moonraker `print_stats.state` | `standby`, `idle` | `IDLE_NO_ACTIVE_PRINT` |
| Moonraker `print_stats.state` | `complete`, `cancelled` | `TERMINAL` |
| Moonraker `print_stats.state` | `error` | `ERROR` |
| Bambu MQTT status | (adapter maps equivalents) | Same enums |
| Any/unknown | other or missing | `UNKNOWN` |

### 4.6 Klipper/Moonraker Adapter

- **REST API**: File upload, print start, G-code commands (`SET_GCODE_OFFSET`).
- **WebSocket**: Real-time `print_stats` state updates, object queries.
- **Compatibility mode**: Shim for Elegoo-modified Moonraker payload variants. Applied before normalization.
- **Command queue**: Priority queue tuple `(priority, enqueue_seq, command)`.
  - Emergency cancel bypasses queue via dedicated session.
  - On job purge, pending stale commands are dropped and in-flight task cancellation is attempted.

**WebSocket reconnection policy**:
- Exponential backoff with jitter: 1s → 2s → 4s → 8s, capped at 30s.
- Full state poll on reconnect (`get_printer_state()`) to reconcile agent vs printer state.
- Handle prints that started or completed during disconnection:
  - If printer moved from IDLE to PRINTING during disconnect → trigger external print detection.
  - If printer moved from PRINTING to TERMINAL during disconnect → trigger cleanup.
  - If state is UNKNOWN → stay disconnected and notify user.

### 4.7 Bambu Adapter (MVP-Minimal)

- **Transport**:
  - Upload artifacts via local FTP/FTPS.
  - Status + control events via local MQTT broker.
- Implements the same `PrinterAdapter` interface.
- **MVP command set**: `start_print`, `pause_print`, `resume_print`, `cancel_print`, `emergency_cancel`, `get_printer_state`.
- Z-offset commands (`get_z_offset`, `set_z_offset`) return `NotSupported` in MVP; Bambu prints are NOTHING-only monitoring until full adapter parity in Phase 2.
- Safety policy mirrors Klipper side: emergency stop uses adapter-specific high-priority MQTT channel.
- Reconnection: MQTT client auto-reconnect with backoff. State poll on reconnect.

### 4.8 Slicer Pipeline

- Accept `.stl` (MVP) and `.step` (Phase 2) intake flow.
- Run headless PrusaSlicer or OrcaSlicer CLI and parse:
  - `FirstLayerParams`: layer height, nozzle temp, bed temp, speed.
  - `FirstLayerBounds`: XY bounding box of first-layer moves from G-code parsing.
- **Proactive geometry analysis**: detect overhangs >60°, bridges, thin walls. Agent recommends optimizations to user (e.g., "Detected overhangs >60° — enable tree supports? Estimated +15g filament, +20min print time") rather than silently slicing with defaults.
- If extraction fails, print may proceed with NOTHING-only monitoring.
- For external prints, metadata parsing is best-effort and used for observability only; it does not enable first-layer actuation.

### 4.9 Vision Monitor

#### 4.9.1 Camera Calibration Lifecycle

**Method**: Checkerboard pattern placed at bed level. User-triggered via `/calibrate_camera` Telegram command.

**Stored artifacts** (persisted to `data/calibration/`):
- Homography matrix
- `mm_per_pixel` value
- Capture timestamp
- Camera identifier

**Validity criteria**:
- Expires after configurable period (default: 30 days) or on reported camera position change.
- Validity checked before any first-layer correction cycle.

**Degraded behavior** when calibration is stale or missing:
- First-layer correction disabled.
- NOTHING detection remains active (does not require calibration).
- User notified: "Camera calibration expired/missing. First-layer correction disabled. Run /calibrate_camera."

#### 4.9.2 NOTHING Detection

- NOTHING
- Temporal confirmation windows:
  - **High confidence**: 3/5 frames > 0.85 → emergency cancel.
  - **Moderate confidence**: 2/3 frames 0.6-0.85 → pause + ask user.
- Active for both agent-started and external prints.

#### 4.9.3 First-Layer "God Hand" Classifier

- OpenCV-based analysis of calibrated ROI.
- Visual classification logic: gaps between extrusion lines → Z too high; filament scraping/ridging → Z too low.
- Correction via `SET_GCODE_OFFSET Z_ADJUST` (±0.01mm step, ±0.05mm clamp).
- Only executes when all preflight gates pass:
  - Agent-started job (not external).
  - Valid camera calibration.
  - `FirstLayerParams` and `FirstLayerBounds` available.
- Coverage guard skips low-signal ROI cycles.
- Evaluation cadence: 5s with 2-consecutive hysteresis and 10s cooldown.

### 4.10 Audio Monitor (Phase 2)

- YAMNet-based audio classifier on microphone input (USB mic or IP camera audio stream).
- Target anomaly classes:
  - **Stepper motor skipping**: rhythmic clicking/thudding indicating missed steps or mechanical binding.
  - **Glass bed cracking**: sharp crack sound indicating thermal stress failure.
  - **Filament grinding**: continuous grinding/clicking at extruder indicating filament jam or stripped gear.
- On detection of mechanical safety events, agent triggers `M112` (firmware emergency stop) immediately — does not wait for vision confirmation.
- Audio detection is independent of and additive to vision-based NOTHING detection.

### 4.11 Filament Spool Monitor (Phase 2)

- Secondary USB camera pointed at filament spool holder.
- **Volume estimation**: estimate remaining filament diameter/weight from spool image to predict runout.
- **Tangle detection**: visual detection of crossed filament lines on spool surface — a common failure mode invisible to filament runout sensors. Tangles cause mid-print jams when the crossed section reaches the extruder.
- On tangle detection or low-filament warning, agent pauses print and notifies user.

### 4.12 Calibration Automation (Phase 2)

- Agent generates temperature tower and flow-rate tower G-code for specified filament.
- Agent prints towers and uses vision to score each zone (surface quality, stringing, layer adhesion).
- Agent proposes optimal values to user.
- On approval, agent injects values into Klipper config via Moonraker `server/config/save` API and executes `SAVE_CONFIG` G-code to persist across restarts.

### 4.13 Image-to-3D Pipeline (Phase 3)

- Accept photo or sketch via Telegram.
- Call external Image-to-3D API (TripoSR, Meshy, Tripo3D, or self-hosted open-source model).
- **Mesh repair pipeline** (headless MeshLab/Blender scripts):
  - Manifold repair (close holes, fix normals, remove degenerate faces).
  - Thin-wall thickening (ensure minimum wall thickness for FDM printability).
  - Orientation optimization (minimize support volume, maximize bed adhesion surface).
  - Support analysis (estimate support requirements, inform user).
- **Printability gate**: reject meshes that cannot be made structurally sound after repair.
- Output: repaired STL → enters Phase 1 slicer pipeline (§4.8).

### 4.14 LLM Chat with Tool Use

- Anthropic SDK-based async client (`llm/client.py`) with retry on transient errors.
- Per-logical-user `ChatSession` (`llm/chat.py`) with shared Telegram/dashboard history, conversation summary, and safe compaction that preserves recent turns.
- `ChatManager` runs an agentic tool-use loop: LLM calls tools, observes results, continues until text response or max 5 iterations.
- Server-provided `ChatRequestContext` (`llm/runtime.py`) carries request identity, channel, channel-local user/chat IDs, active printer, and request ID into the prompt/tool layer.
- `PrinterContextCollector` (`llm/context.py`) builds the system prompt with live printer state, telemetry, job context, request context, and compacted session summary. System prompt explicitly warns that background context may be stale and tools remain authoritative.
- Tool execution is audit-logged to `data/tool_audit.jsonl` with request ID, logical user, channel, tool name, input, timing, outcome, and whether image content was returned.
- High-risk actions use approval tokens. Raw G-code is default-deny except for safe query-style commands and any explicitly configured allowlist entries (`llm.allowed_gcode_commands`). Shell commands, destructive printer actions, macro execution, restarts, and high-temperature settings can return `approval_required` and must be replayed with the issued approval token.
- Background shell tasks are supported for non-blocking execution. The runtime stores in-memory task state, emits progress/completion callbacks through Telegram or dashboard alerts, and exposes task state for later inspection/cancellation.
- The recommended Anthropic-compatible deployment model is `volcengine-anthropic-doubao-seed-20-pro`, which supports direct image input and image-bearing `tool_result` responses through the current proxy.
- Tool set in `llm/tools.py` includes:
  - **Printer query**: `get_printer_state`, `get_current_position`, `get_print_progress`, `get_z_offset`, `list_files`, `get_file_metadata`, `get_system_info`, `list_macros`
  - **Printer control**: `start_print`, `pause_print`, `resume_print`, `cancel_print`, `set_nozzle_temp`, `set_bed_temp`, `home_axes`, `set_fan_speed`, `set_z_offset`, `move_axis`, `extrude`, `motors_off`, `set_speed_override`, `set_flow_override`, `run_gcode`, `execute_macro`, `restart_firmware`, `restart_service`
  - **Camera**: `camera_snapshot`, `camera_move_step`, `camera_move_to`, `camera_get_presets`, `camera_go_to_preset`, `camera_get_position`
  - **Runtime ops**: `run_shell`, `get_background_tasks`, `cancel_background_task`, `get_pending_approvals`
- Most printer-oriented tools accept an optional `printer` parameter. When omitted, routing defaults to the request context’s `active_printer`.
- Safety: nozzle temp capped at 300C, bed at 120C, M112 blocked via run_gcode (must use emergency_cancel path).
- `camera_snapshot` returns multimodal content blocks (JPEG image + text), persists the snapshot under `data/camera_snapshots/`, and can deliver the photo back to Telegram when the request originated from Telegram.
- `run_shell` is intended to reduce long-term pressure to add one-off custom tools. Low-risk inspection commands may run directly; high-risk commands return an approval-required result instead of executing.

### 4.15 Camera PTZ Control

- `TapoCamera` (`camera/tapo.py`) uses `python-onvif-zeep` for ONVIF protocol.
- ONVIF connection via `asyncio.to_thread` (blocking ONVIF SDK calls offloaded).
- Supports: move_step (left/right/up/down), move_to (absolute pan/tilt), get_presets, go_to_preset, and best-effort `get_position()` via ONVIF status when the camera reports PTZ coordinates.
- Step movements use velocity-based continuous move with 0.3s duration + stop.
- Configuration via `TapoCameraConfig` (host, user, password, onvif_port).
- PTZ hard-limit detection is not inferred blindly. If the camera API cannot report limit state, tool responses surface `limit_status: unknown`.

### 4.16 Web Dashboard API

- `aiohttp`-based web server (`api/server.py`) with REST + WebSocket endpoints.
- REST: `/api/health`, `/api/printers`, `/api/status`, `/api/telemetry/{printer_id}`, `/api/actions/{printer_id}/{action}`, `/api/llm/chat`, `/api/v1/chat/context`, `/api/v1/tool-audit`, `/api/v1/background-tasks`, `/api/v1/pending-approvals`.
- WebSocket: `/ws` for real-time telemetry push to dashboard clients.
- MJPEG camera proxy: `/api/camera/stream` converts RTSP to browser-compatible MJPEG.
- Optional API token authentication (`DashboardConfig.api_token`).
- Static file serving for dashboard UI at `/`.
- Dashboard chat uses the same logical-user session model as Telegram instead of a hardcoded `user_id=0`.

---

## 5. Data Flow

### Agent-Started Print (Happy Path)

```
User -> Telegram: .stl file
  -> Bot/Auth: validate user allowlist
  -> Bot: stage file, ACK to user
  -> Slicer Pipeline: headless slice + metadata extraction
  -> Slicer Pipeline: geometry analysis (overhangs, bridges, thin walls)
  -> Bot: send summary + proactive recommendations + Approve/Reject inline keyboard
  -> Decision Engine: await user response (10min timeout)
  -> Adapter: upload_file() + start_print()
  -> State Machine: PRINTING -> MONITORING
  -> Vision Monitor: NOTHING detection (continuous)
  -> Vision Monitor: first-layer "God Hand" classifier (first N layers)
  -> Z-Offset Controller: SET_GCODE_OFFSET Z_ADJUST bounded corrections
  -> (print completes)
  -> Z-Offset Controller: restore baseline + verify
  -> State Machine: COMPLETE -> IDLE
  -> Bot: notify user with result
```

### External Print Detection

```
Adapter WebSocket: print_stats.state -> "printing" (debounced)
  -> State Machine: IDLE -> MONITORING
  -> Vision Monitor: NOTHING-only (no first-layer correction)
  -> (print completes or fails)
  -> State Machine: COMPLETE/FAILED -> IDLE
```

### Photo-to-Print (Phase 3)

```
User -> Telegram: photo or sketch
  -> Bot/Auth: validate user allowlist
  -> Image-to-3D API: generate mesh (TripoSR/Meshy/Tripo3D)
  -> Mesh Repair: MeshLab/Blender headless (manifold, walls, orientation)
  -> Printability Gate: reject if structurally unsound
  -> Bot: send 3D preview + print estimates + Approve/Reject
  -> [enters Agent-Started Print flow from Slicer Pipeline step]
```

---

## 6. Security

### Telegram Authentication

- **Allowlist**: Authorized user IDs or chat IDs configured at deployment in config file.
- **Rejection**: Unauthorized messages receive a brief error reply (no silent drop — aids debugging).
- **Group chat**: Not supported in MVP. Bot operates in private chat mode only.
- **Scope**: All commands and file uploads require authorization. No public-facing endpoints.

### Secrets

- Bot token, printer credentials stored in environment variables or `.env` file (gitignored).
- No hardcoded secrets in source code.

---

## 7. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| One Agent Core per printer | Single-instance MVP | Simplifies state machine, avoids concurrency across printers. Multi-printer coordination deferred to Phase 3. |
| JSON over SQLite for MVP state | Atomic JSON snapshots | Simpler, no DB dependency. SQLite added in Phase 2 for history/analytics. |
| External prints are NOTHING-only | No first-layer correction | Agent cannot verify calibration state or G-code origin for external prints. Safety first. |
| Dedicated emergency cancel path | Bypasses command queue | Emergency stop must not wait behind queued commands. Uses separate connection. |
| Rule-based safety, not LLM | Local deterministic logic | Real-time safety decisions cannot tolerate LLM latency or hallucination risk. |
| Per-job decision lock | asyncio.Lock keyed by job_id | Eliminates timeout-vs-user race conditions across all decision entry points. |
| RTSP for camera input | Network IP camera support | User's existing cameras (Blurams, Wyze) are IP cameras with RTSP streams. |
| Bambu adapter in MVP (minimal) | start/pause/cancel/status | User owns 3 Bambu printers. Minimal support enables unified workflow; full parity deferred. |

---

## 8. Reference Implementation Notes (OpenClaw)

[OpenClaw](https://github.com/openclaw/openclaw) is an open-source AI assistant with Telegram integration and a community Snapmaker 3D-printer skill. The patterns below are worth adopting, adapted from TypeScript to Python asyncio.

### 8.1 Moonraker API Client

**Source**: [`skills/lucakaufmann/snapmaker/scripts/snapmaker.py`](https://github.com/openclaw/skills/tree/main/skills/lucakaufmann/snapmaker)

The Snapmaker skill contains a working Moonraker REST client with useful field mappings:

| Moonraker Query Field | Usage |
|----------------------|-------|
| `print_stats.state` | Job state (`standby`, `printing`, `paused`, `complete`, `error`) |
| `print_stats.filename` | Current file |
| `print_stats.print_duration` | Elapsed seconds |
| `print_stats.info.current_layer` / `.total_layer` | Layer progress |
| `virtual_sdcard.progress` | Raw progress (0.0-1.0) |
| `display_status.progress` | Layer-aware progress (more accurate) |
| `extruder.temperature` / `.target` | Nozzle temps |
| `heater_bed.temperature` / `.target` | Bed temps |

**Adaptation for Sensing Voxel**:
- Replace raw socket HTTP with `aiohttp.ClientSession` for async.
- Add WebSocket subscription for real-time `print_stats` events (OpenClaw only uses REST polling).
- Add normalized state mapping per our §4.5 contract.
- Add reconnection logic (OpenClaw's skill exits on failure instead of retrying).

### 8.2 Retry with Exponential Backoff

**Source**: [`src/infra/retry.ts`](https://github.com/openclaw/openclaw/blob/main/src/infra/retry.ts), [`src/infra/backoff.ts`](https://github.com/openclaw/openclaw/blob/main/src/infra/backoff.ts)

OpenClaw's generic retry framework computes delay as:

```
delay = min(max_delay, base_delay * factor^attempt)
jitter = delay * jitter_factor * random()
final  = delay + jitter
```

Default backoff policy: `initial=2s, max=30s, factor=1.8, jitter=0.25`.

**Adoption**: Implement as a shared `retry_async()` utility in `src/sensing_voxel/infra/retry.py`. Used by Moonraker WebSocket reconnection (§4.6), Bambu MQTT reconnection (§4.7), and Telegram send retries.

### 8.3 WebSocket/Stream Reconnection Loop

**Source**: [`src/signal/sse-reconnect.ts`](https://github.com/openclaw/openclaw/blob/main/src/signal/sse-reconnect.ts)

Cleanest reconnection pattern in the codebase:

```
while not shutdown:
    try:
        await connect_and_stream()
        # stream ended normally
    except recoverable_error:
        pass
    reconnect_attempts += 1
    delay = compute_backoff(policy, reconnect_attempts)
    await sleep_or_shutdown(delay)
    # KEY: reset reconnect_attempts to 0 on first successful event
```

**Adoption**: Direct model for our Moonraker WebSocket reconnection (§4.6) and Bambu MQTT reconnection (§4.7). In Python asyncio, use `asyncio.Event` for shutdown signaling instead of `AbortSignal`.

### 8.4 Telegram Interaction Patterns

**Source**: [`src/telegram/model-buttons.ts`](https://github.com/openclaw/openclaw/blob/main/src/telegram/model-buttons.ts), [`src/telegram/send.ts`](https://github.com/openclaw/openclaw/blob/main/src/telegram/send.ts), [`src/telegram/network-errors.ts`](https://github.com/openclaw/openclaw/blob/main/src/telegram/network-errors.ts)

Three patterns to adopt:

1. **Inline keyboard callback_data protocol**: Prefix-based encoding to stay within Telegram's 64-byte limit. E.g., `apr_{job_id[:8]}` for approve, `rej_{job_id[:8]}` for reject, `res_{job_id[:8]}` for resume, `can_{job_id[:8]}` for cancel. Parse with a discriminated union / dataclass pattern.

2. **Unified send function**: Single function handles text, photo, and document sending with type detection. Fallback: if `sendPhoto` fails for large images, retry as `sendDocument`.

3. **HTML render with fallback**: Send as Telegram HTML; if parse fails, auto-retry as plain text. Eliminates formatting-related send failures.

4. **Recoverable error classification**: Classify errors by `errno` code (`ECONNRESET`, `ETIMEDOUT`, etc.) and message keywords (`timeout`, `reset`, `unavailable`). Traverse error chain. Only retry on recoverable errors.

### 8.5 File-Based State Persistence with Locking

**Source**: [`src/config/sessions/store.ts`](https://github.com/openclaw/openclaw/blob/main/src/config/sessions/store.ts), [`src/agents/session-write-lock.ts`](https://github.com/openclaw/openclaw/blob/main/src/agents/session-write-lock.ts)

OpenClaw uses JSON file persistence with:
- In-memory TTL cache (45s) with file mtime invalidation.
- File-based write lock with PID liveness check and stale lock timeout.
- Signal handler cleanup (`SIGTERM`, `SIGINT`) for lock release on crash.

**Adaptation**: Our `persistence.py` already uses atomic temp-file + rename. Add `fcntl.flock()` or the `filelock` library for write locking, and `signal` handlers for cleanup. The TTL cache is unnecessary for single-instance MVP but useful if expanded later.

### 8.6 File Staging and Manifest

**Source**: [`skills/ajmwagar/find-stl/scripts/find_stl.py`](https://github.com/openclaw/skills/tree/main/skills/ajmwagar/find-stl)

The find-stl skill stages downloaded files with:
- Structured directories: `~/models/incoming/{source}-{id}-{slug}/files/`
- `manifest.json` tracking source URL, attribution, license, file list with SHA-256 hashes.
- `safe_slug()` for filesystem-safe naming.

**Adoption**: Apply to our slicer pipeline — track uploaded STL, generated G-code, and print metadata in a job manifest. Useful for job history when SQLite is added in Phase 2.

### 8.7 Not Adopted

| OpenClaw Pattern | Reason Skipped |
|-----------------|----------------|
| Multi-platform chat (Discord, Signal, iMessage) | Only Telegram needed |
| LLM-centric skill system (prompt injection) | Our logic is rule-based, not LLM-driven |
| Grammy framework internals | Using `python-telegram-bot` instead |
| Community skill sharing (ClawhHub) | Single-purpose agent |
| Multi-session management | Shared per-logical-user chat session across Telegram and dashboard |
| Disk budget / transcript archival | Overkill for MVP |

---

## 9. Implementation Status (as of 2026-03-27)

This section tracks actual implementation state vs. the architecture spec.

### Implemented (Core Infrastructure)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Core data models | `core/models.py` | Complete | All enums, dataclasses, and event types per spec |
| Agent state machine | `core/state_machine.py` | Complete | Transition table enforced, hook rollback implemented |
| Decision engine | `core/decision.py` | Complete (1 test failing) | Lock-guarded CAS correct; WeakValueDictionary causes 1 failing test |
| State persistence | `core/persistence.py` | Complete | Uses `filelock.SoftFileLock` (cross-platform) |
| Retry infrastructure | `infra/retry.py` | Complete | Correct backoff with jitter; `retry_async` and `retry_async_with_reset` |
| Printer adapter ABC | `adapters/base.py` | Complete | All abstract methods, event emitters, `NotSupportedError` |
| Command queue | `adapters/command_queue.py` | Complete | Priority queue, emergency bypass, 0.5s shutdown poll timeout |
| Klipper/Moonraker adapter | `adapters/klipper.py` | ~85% | REST+WS, base_url support, telemetry parsing, state normalization |
| Bambu adapter | `adapters/bambu.py` | ~80% | MQTT+FTP, start/pause/cancel/status, emergency cancel |
| Telegram bot | `bot/telegram.py` | ~70% | Auth, upload, decisions, DI, notifications; missing AI chat wiring |
| Configuration | `config.py` | Complete | All config sections including LLM, TapoCamera, Dashboard |
| Entry point | `main.py` | Complete | Async main, multi-printer, dashboard, LLM wiring, chat photo sender, default logical user |
| Web dashboard API | `api/` | Complete | REST endpoints, WebSocket, MJPEG proxy, token auth, chat context and audit endpoints |
| Dashboard UI | `dashboard/static/` | ~40% | Real-time telemetry, multi-printer selector |
| LLM client | `llm/client.py` | Complete | Async Anthropic SDK wrapper with retry |
| LLM runtime | `llm/runtime.py` | Complete | Request context, token estimation, compaction helpers, tool audit logger, approvals, background tasks |
| LLM chat | `llm/chat.py` | Complete | Shared logical-user sessions, auto-compaction, context diagnostics, audit access, task/approval visibility |
| LLM context | `llm/context.py` | Complete | Printer context collector + request context + summary-aware system prompt builder |
| LLM tools | `llm/tools.py` | Complete | Printer control/query, camera, shell, printer targeting, audit-aware executor |
| Camera PTZ | `camera/tapo.py` | Complete | ONVIF-based Tapo C220 pan/tilt control + best-effort position reporting |
| Monitoring orchestrator | `monitoring/monitor.py` | ~60% | External print detection done, NOTHING routing done |

### Not Yet Implemented

| Component | Files | Blocking Tasks |
|-----------|-------|----------------|
| Slicer pipeline | `slicer/pipeline.py`, `slicer/metadata.py` (stubs) | TASKS.md 1.10 |
| NOTHING detection | `vision/NOTHING.py`, `vision/camera.py` (stubs) | TASKS.md 1.11 |
| Camera calibration | `vision/calibration.py` (stub) | TASKS.md 1.12 |
| First-layer classifier | `vision/first_layer.py` (stub) | TASKS.md 1.13 |
| Z-offset controller | `monitoring/z_offset.py` (stub) | TASKS.md 1.13 |
| Integration tests | `tests/integration/` (sparse) | TASKS.md 1.16 |

### Known Deviations from Architecture Spec

| Spec Reference | Spec Requirement | Actual Implementation | Fix Required |
|---|---|---|---|
| §8.2 | Jitter should be random | Uses `event_loop.time() % 1` (deterministic) | Yes — BUG-REV-4 |
| §4.6 | Compatibility shim for Elegoo/Neptune | `compatibility_mode` bool stored but shim not applied | Partial — needs implementation |
| §6 | Secrets in env vars or `.env` (gitignored) | Config reads from YAML; no env var fallback | Low risk for MVP |
| Runtime plan | First-class reminder/scheduler helpers | Background execution exists and can support sleep-based reminders, but there is no dedicated milestone/reminder planner yet | Planned follow-up |
