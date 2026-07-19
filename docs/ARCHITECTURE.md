# Sensing Voxel — Architecture

**Version**: 2.0
**Date**: 2026-07-18
**Status**: Current runtime architecture

---

## 1. System Overview

Sensing Voxel is a Python 3.12 asynchronous service that combines authenticated user interfaces, deterministic print authorization, printer adapters, LLM-assisted operations, and camera-based print monitoring.

`src/sensing_voxel/main.py` is the composition root. It loads configuration, creates printer adapters, creates one `PrintSafetyService` for each configured printer, attaches available `PrintMonitor` instances, and starts the configured Telegram, dashboard, and LLM services.

```text
Telegram client                 Browser client
       |                              |
Telegram authorization       HTTP + WebSocket API
       |                              |
       +---------- Agent Core --------+
                    |
          PrintSafetyService[printer]
          |       |        |        |
      artifact validator preview preflight
                    |
              PrinterAdapter
                    |
           Klipper / Moonraker
                    |
               PrintMonitor
      camera snapshot + isolated LLM review
```

The runtime separates two paths:

1. **Deterministic authorization** — content identity, whole-file semantic validation, deterministic previews, physical confirmation, approval binding, live preflight, and adapter execution.
2. **Probabilistic observation** — periodic camera snapshots analyzed by an image-capable LLM, followed by configured alert and pause behavior.

Camera or LLM output never replaces deterministic authorization.

---

## 2. Runtime Packages and Ownership

```text
src/sensing_voxel/
├── main.py                  # Runtime composition and lifecycle
├── config.py                # Configuration models and loading
├── core/                    # State machine, decisions, persistence, shared models
├── safety/                  # Artifacts, semantic validation, previews, authorization
├── monitoring/
│   └── print_monitor.py     # Canonical camera/LLM print monitor
├── adapters/                # Printer contract and transport implementations
├── bot/                     # Telegram authorization, handlers, and decisions
├── api/                     # Authenticated REST and WebSocket dashboard API
├── dashboard/               # Browser assets
├── llm/                     # Chat, tools, approvals, context, tasks, and audit
├── camera/                  # ONVIF PTZ integration
├── slicer/                  # Model processing support; not fully runtime-wired
├── storage/                 # SQLite repositories used by wired services
└── infra/                   # Shared retry and G-code utilities
```

Each package has one primary responsibility. `PrintSafetyService` is the sole final-G-code authorization owner, and `PrintMonitor` is the sole print-camera monitoring owner. Product expansion should extend these boundaries rather than introduce parallel validation, camera-capture, monitoring, or orchestration paths.

---

## 3. Safety-Managed Print Flow

```text
authenticated G-code intake
  -> classify source and select target printer
  -> retain exact bytes under a SHA-256 identity
  -> interpret every executable line under the supported semantic model
      -> reject any blocking or unsupported behavior
  -> produce deterministic summary JSON and rendered SVG preview
  -> collect material/print metadata and a configured/live physical snapshot
  -> obtain explicit operator approval of the bound evidence
  -> bind artifact, validation, previews, printer, configuration, source, and user
  -> re-read live printer, configuration, and Z facts
      -> reject stale, unavailable, mismatched, or unsafe facts
  -> validate and start the required monitor contract
      -> fail closed if readiness or startup fails
  -> re-read live printer and Z facts again
      -> reject any change during monitor startup
  -> start the exact printer-side artifact through the selected adapter
  -> return the execution result
```

The ordering is an invariant: review precedes approval; approval precedes the first live preflight; mandatory monitor readiness and startup precede a second live preflight; printer start occurs only after both preflights pass.

### 3.1 Content-Addressed Artifact

The safety service retains exact source bytes under their SHA-256 digest. Validation, previews, approval, and execution refer to that identity rather than trusting a mutable file path.

The current store is in memory. Its data does not survive a process restart, so a restarted runtime requires a fresh authorization cycle.

### 3.2 Whole-File Semantic Validation

The validator interprets every executable line in order while maintaining relevant machine state. It:

- enforces supported command and parameter allowlists;
- tracks absolute and relative positioning and extrusion modes;
- checks homing, heating, motion, extrusion, feed, coordinate, and thermal semantics;
- derives toolpath and first-layer envelopes and first-layer Z where available; and
- emits structured issues and a deterministic result digest.

Unknown safety-relevant behavior is rejected rather than guessed. This is a conservative model of supported G-code, not firmware emulation, collision simulation, or proof of a successful print.

### 3.3 Deterministic Preview

A valid artifact produces canonical summary JSON and an SVG rendering of the interpreted extrusion path. The source, validation result, summary, and SVG each contribute to the approval binding.

Rendering is bounded to 20,000 extrusion segments and 2,000,000 SVG bytes. The output is an approval aid, not a physical simulation.

### 3.4 Physical Snapshot and Approval

The request combines G-code material/print metadata with a `PhysicalConfirmationSnapshot` derived from the configured build-plate memo, hotend/nozzle configuration, and live probe Z offset. The operator reviews and explicitly approves this evidence. The resulting binding identifies the exact target, artifact, previews, validation, evidence, source, and user.

This gives the operator configured and live evidence, but cannot independently prove bed clearance, correct material, hardware integrity, or safe surroundings.

### 3.5 Live Printer and Z Preflight

Immediately before monitor startup, the service re-reads available target identity, hardware/profile/configuration/macro fingerprints, source trust, user identity, probe Z offset, and homing-origin Z. After the mandatory monitor passes readiness checks and starts, the service repeats the live printer/Z preflight before starting the file. Missing required facts or drift from the approved snapshot fails closed.

The enforced Z sanity barriers block an uninitialized zero probe offset and a positive homing-origin Z. The service also rejects disconnected targets and adapter errors rather than converting them to warnings.

### 3.6 Execution

Only after approval, both preflights, and mandatory monitor startup pass does the service start the exact printer-side file through the selected `PrinterAdapter`. Partial adapter failure is not reported as success; reconciliation and operator inspection may still be required if a remote side effect occurred before a later failure.

---

## 4. Mandatory Print Monitoring

Safety-managed prints classified as `user_gcode` or `sva_generated` require a monitor contract containing the printer-side filename, source identity, and validated first-layer Z. Before the monitoring loop starts, readiness checks require usable printer progress, a camera snapshot with the printer visible, and a conclusive independent LLM analysis. Missing contract data, failed readiness, or monitor startup failure makes the managed start fail closed.

### 4.1 Cadence and Observation

```text
first layer: every 60 seconds
  -> obtain camera snapshot
  -> analyze in an independent LLM context
  -> classify visible anomaly evidence
  -> apply acknowledgement and pause policy
  -> publish configured notification or alert

after first layer: use configured normal interval
```

The independent analysis context prevents chat history from biasing an observation. An unacknowledged detected anomaly causes an automatic printer pause.

Monitoring remains probabilistic. Lighting, framing, focus, occlusion, stream failure, latency, model capability, and model error can cause false positives or false negatives. It does not change Z offset, and a 60-second cadence is not continuous hazard detection.

---

## 5. Printer Adapter Boundary

`PrinterAdapter` normalizes printer state, telemetry, position, upload/start, lifecycle, and command operations.

### 5.1 Klipper/Moonraker

The Klipper adapter is the production-supported path. It uses Moonraker REST for queries and commands and WebSocket updates for live state and telemetry. Moonraker status responses are normalized from `result.status.{object}` where applicable.

Normal hardware work is serialized. Emergency cancellation uses a dedicated session so it does not wait behind ordinary commands. Printer-configuration replacement must create a backup before writing `printer.cfg`.

### 5.2 Bambu

The Bambu adapter provides local MQTT and file-transfer behavior, but the default runtime does not wire it as a production-equivalent path. It requires deployment validation and equivalent safety and monitoring guarantees before parity can be claimed.

---

## 6. Agent Core, Decisions, and Persistence

The Agent Core coordinates orchestration state, interfaces, and user decisions. It does not supersede the target printer's safety service.

Decision consumption uses lock-guarded compare-and-set behavior. The runtime validates the current state, job, token, and prior-consumption status before persisting a consumed decision and releasing the lock for external effects. This prevents callback, timeout, recovery, or duplicate-request races from executing the same decision twice.

Agent state uses versioned JSON with locking and atomic replacement. SQLite repositories store wired data such as printer memos. These persistence paths are distinct from the process-local safety artifact store.

The LLM subsystem provides per-user chat context, printer context, tools, approval-token replay for configured high-risk actions, background work, and JSONL audit records. Its suggestions and tool results remain subordinate to adapter checks and deterministic print authorization.

---

## 7. Interface and Security Boundaries

### 7.1 Telegram

Telegram handlers enforce the configured user allowlist before accepting commands or uploads. Decisions use the same idempotent consumption rules as other interfaces.

### 7.2 HTTP API

The dashboard uses `aiohttp`. `/health` is the only public path; every other HTTP route requires valid bearer authentication and fails closed when authentication is absent or invalid.

Startup rejects a dashboard bind outside `127.0.0.1`, `::1`, or `localhost` when no API token is configured.

### 7.3 WebSocket

WebSocket connection setup separately enforces configured token and allowed-origin policy.

### 7.4 Deployment Boundary

Application authentication does not terminate TLS or provide host firewalling, VPN access, network segmentation, secret rotation, or operating-system hardening. Those remain deployment responsibilities.

---

## 8. Current Limitations and Evolution

- The safety artifact and approval data are process-local.
- Supplied G-code is the complete safety-managed workflow; model-file slicing is not fully wired through Telegram and dashboard intake.
- Camera/LLM monitoring can miss failures or pause healthy prints.
- Physical confirmation presents configured and live evidence for operator review, and preflight sees only exposed, modeled facts.
- Klipper/Moonraker is the production-supported printer path.
- Durable multi-printer scheduling is absent; job-history persistence and API views are wired.
- OpenAI-compatible providers do not yet preserve the full tool-use loop required by this runtime.
- Software controls cannot prevent every electrical, thermal, mechanical, firmware, material, network, environmental, or operator failure.

Future outcomes may include better model preparation, material guidance, maintenance trends, and job scheduling. They should be implemented by extending the canonical safety, monitoring, adapter, and persistence boundaries rather than adding overlapping subsystems.

---

## 9. Architectural Invariants

- A safety-managed artifact starts only through its target printer's `PrintSafetyService`.
- Every executable G-code line is interpreted under the supported semantic model.
- Unknown safety-relevant syntax fails closed.
- Approval binds exact source and derived evidence, not a mutable filename alone.
- Material/print metadata and the configured/live physical snapshot are explicit approval evidence.
- Mutable printer and Z facts are checked before monitor startup and again before printer start.
- Required sources cannot report a successful managed start without a complete monitor contract and successful monitor startup.
- Camera and LLM output never authorize G-code or actuate Z offset.
- `/health` is the dashboard API's only public route.
- Non-loopback dashboard exposure requires an API token.
- A consumed decision causes side effects at most once.
- Emergency cancellation does not wait behind normal printer work.
