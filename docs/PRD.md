# Sensing Voxel - Product Requirements Document

**Version**: 1.0
**Date**: 2026-02-24
**Status**: Approved
**Source**: `.design/plan.md` v10 + `.design/review.md` fixes

---

## 1. Overview

### 1.1 Vision

Sensing Voxel is an AI print-operations agent that minimizes manual 3D-printing work from file intake through print monitoring and recovery actions. It combines Telegram-driven job intake, automated slicing, camera/audio monitoring, and printer control across a mixed hardware fleet (Klipper/Moonraker and Bambu) so users can start, monitor, and recover prints with minimal intervention.

### 1.2 Problem Statement

Current home and prosumer 3D-print workflows are fragmented and manual: users manually slice, tune first layer, watch for NOTHING failures, and inspect calibration towers by eye. This leads to failed overnight prints, wasted filament, and inconsistent quality. The problem is worse in mixed-printer environments where control APIs differ (Moonraker vs Bambu protocols), forcing duplicated manual processes.

### 1.3 Goals & Success Metrics

| Goal | Metric | Target |
|------|--------|--------|
| Reduce failed prints | Failure rate on monitored jobs | < 5% |
| Detect defects early | Time from defect onset to pause/cancel | < 30s |
| Reduce manual workflow | User interactions from file to print start | <= 3 |
| First-layer quality | First-layer classification accuracy (qualified conditions) | > 80% |
| Safety race correctness | Duplicate decision actions (approve/resume/timeout races) | 0 in test suite |
| External print coverage | External active prints detected while agent idle | 100% |

---

## 2. Users & Personas

### 2.1 Target Users

- Hobbyists and makers running Klipper printers (Ender-class, Neptune-class, custom builds).
- Users with Bambu printers who want a unified Telegram-first workflow.
- Small-batch operators managing 2-6 printers and needing unattended monitoring.

### 2.2 Persona: Wei - Mixed-Fleet Hobby Operator

- **Background**: Runs an Ender 3 V2 (Klipper/Fluidd), Neptune 4 Max (modified Moonraker/Fluidd), and Bambu P1-series printer.
- **Goals**: Submit models from phone, avoid babysitting, get consistent first layers and fewer failed long prints.
- **Pain Points**: Different printer control surfaces, manual slicer steps, late failure discovery, and repetitive tuning work.

---

## 3. Use Cases

### UC-01: Telegram File-to-Print Workflow

**Actor**: Wei
**Trigger**: Wei sends `.stl` to Telegram bot.
**Flow**:
1. Agent validates file type/size and stages file.
2. Agent runs headless slicing and geometry checks (overhang >60°, bridges, thin walls).
3. Agent sends summary with recommendations and inline **Approve/Reject** decision.
4. On approve, agent uploads G-code and starts print.
5. On timeout (default: **10 minutes**, configurable) or reject, agent performs local cleanup and notifies user.

**Success**: Print starts within 2 minutes of approval, with proactive support recommendations.

### UC-02: First-Layer Auto-Correction (Agent-Initiated Jobs Only)

**Actor**: Agent
**Trigger**: Agent-started print enters first-layer monitoring window.
**Flow**:
1. Verify calibration validity (`homography`, `mm_per_pixel`) and `FirstLayerParams` availability.
2. Compute toolpath-aware ROI from first-layer G-code bounds.
3. Evaluate at 5s cadence with 2-consecutive hysteresis and 10s cooldown.
4. Inject bounded Z-offset corrections via Moonraker (`±0.01mm` step, clamp `±0.05mm`).
5. **Saturation exit**: if cumulative correction reaches clamp boundary and classifier still reports poor quality for 3 consecutive evaluations, pause print and notify user with current offset and classification result.
6. Restore Z-offset baseline on terminal transition.

**Success**: First layer converges without user intervention and no cross-print drift.

### UC-03: NOTHING Detection With Controlled Stop

**Actor**: Agent
**Trigger**: Active print monitoring detects sustained failure pattern.
**Flow**:
1. Vision model emits confidence stream.
2. If high confidence (3/5 frames >0.85), emergency cancel path executes.
3. If moderate confidence (2/3 frames 0.6-0.85), pause and request **Resume/Cancel**.
4. If pause decision times out (default: **5 minutes**, configurable), auto-cancel active job.

**Success**: Defects are stopped quickly with clear user notifications.

### UC-04: External Print Detection (Started Outside Agent)

**Actor**: Agent
**Trigger**: WebSocket `print_stats.state` transitions to `printing` or `paused` while agent is IDLE.
**Flow**:
1. Agent debounces state transition (2 updates / up to 3s).
2. Agent transitions to monitoring mode.
3. Monitoring policy is **NOTHING-only** for the entire external job.
4. Agent may parse layer markers for notification context only.

**Policy**: External prints receive NOTHING-only monitoring. First-layer auto-correction is **never** enabled for external prints, even if metadata can be parsed.

**Success**: External jobs are monitored for catastrophic failures without unsafe first-layer actuation.

### UC-05: Closed-Loop Temperature/Flow Calibration

**Actor**: Wei
**Trigger**: User requests filament calibration.
**Flow**:
1. Agent generates temp/flow towers.
2. Agent scores zones visually and proposes optimal values.
3. On approval, agent writes calibrated values through printer adapter.

**Success**: Reduced manual calibration iterations.
**Scope**: Nice-to-Have (Phase 2).

### UC-06: Predictive Maintenance Tracking

**Actor**: Agent
**Trigger**: Scheduled or user-triggered shaper calibration events.
**Flow**:
1. Collect X/Y resonance values over time.
2. Detect drift threshold breaches.
3. Notify user with action recommendation.

**Success**: Mechanical issues identified before severe print degradation.
**Scope**: Nice-to-Have (Phase 3).

### UC-07: Bambu Printer Interface

**Actor**: Agent
**Trigger**: Job targets Bambu printer.
**Flow**:
1. Upload print assets via local FTP/FTPS.
2. Send job control and receive status via local MQTT broker.
3. Use adapter-normalized state/events so Agent Core logic remains unchanged.

**MVP Scope**: Minimal command set — start, pause, cancel, status query. Full command parity deferred to Phase 2.

**Success**: Same Telegram workflow works across Klipper and Bambu with adapter-specific transport.

---

## 4. Feature Scope

### 4.1 Must Have (MVP)

- [ ] Telegram bot workflow with upload validation, inline decisions, configurable timeout handling, and **user allowlist authentication**.
- [ ] Klipper/Moonraker printer adapter with compatibility mode for older/modified Moonraker APIs (Neptune-class firmware variants).
- [ ] Bambu printer adapter (minimal): start, pause, cancel, status via local MQTT + FTP/FTPS.
- [ ] Shared printer adapter interface contract (abstract base for Klipper and Bambu adapters).
- [ ] STL slicing pipeline with parameter extraction (`FirstLayerParams`, `FirstLayerBounds`) for agent-started jobs.
- [ ] Camera calibration lifecycle: checkerboard calibration, artifact storage, validity checking, `/calibrate_camera` command.
- [ ] First-layer auto-correction for agent-started jobs only, with saturation exit behavior.
- [ ] NOTHING detection with pause/cancel policy and emergency bypass path.
- [ ] Decision idempotency with lock-guarded CAS helper across callbacks/timeouts/recovery.
- [ ] Persistent state snapshots (JSON) with safe IDLE reset policy preserving safety flags.
- [ ] Z-offset baseline capture/restore contract with verification, dirty-state print blocking, and `/reset_z_offset [target_mm]`.
- [ ] External print detection with NOTHING-only monitoring policy.
- [ ] Moonraker WebSocket reconnection with backoff and state reconciliation.

### 4.2 Nice to Have

- [ ] STEP/3MF intake support.
- [ ] Audio anomaly detection (YAMNet).
- [ ] Closed-loop tower scoring and auto-apply profiles (UC-05).
- [ ] Filament spool/tangle monitoring camera.
- [ ] Full Bambu adapter command parity and richer telemetry.
- [ ] SQLite job history and analytics storage.
- [ ] Multi-printer queue scheduling dashboard.
- [ ] Predictive maintenance tracking (UC-06).

### 4.3 Out of Scope

- Non-printing CAD generation.
- Direct firmware modification/forking.
- Fully autonomous cloud-only orchestration that bypasses local safety controls.
- Bambu cloud API fallback (local-only for safety/privacy in MVP).

---

## 5. Technical Constraints

- **Runtime**: Python 3.12 + `asyncio`.
- **Safety**: Real-time safety decisions are rule-based and local. LLM use is advisory only.
- **Command serialization**: All non-emergency printer commands flow through a prioritized queue. Emergency cancel bypasses queue via dedicated client/session.
- **Deployment model**: MVP deploys **one Agent Core instance per printer**. Each instance has its own state machine, Telegram command namespace, and state file. Multi-printer coordination is deferred.
- **State storage**: JSON atomic snapshots for MVP. SQLite deferred to Phase 2 for job history/analytics.
- **Vision inference**: NOTHING

---

## 6. Hardware Environment

### Printers
- **Creality Ender 3 V2**: Klipper + Fluidd on Raspberry Pi.
- **Elegoo Neptune 4 Max**: Elegoo-modified Fluidd + Klipper on MKSPi (requires Moonraker compatibility mode).
- **BambuLab P1SC, A1C, H2C**: Local MQTT + FTP/FTPS interface (no Klipper/Moonraker).

### Cameras
- **Blurams PTZ Dome Security Camera**: IP camera, RTSP/WebRTC stream.
- **Wyze Camera**: IP camera, RTSP stream (via firmware or bridge).

> Note: Vision module must support RTSP network streams. If native support is insufficient, USB webcam alternatives (e.g., Logitech C920/C270) should be considered as fallback.

---

## 7. Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Duplicate decision handling from async races | High | Medium | Lock-guarded `consume_decision(...)` used by all entry points; race tests required |
| External-print policy drift reintroduces unsafe actuation | High | Medium | Single canonical rule: external jobs are NOTHING-only; policy asserted in integration tests |
| `z_offset_dirty` lost during idle reset | High | Low | Full IDLE snapshot preserving persistent fields |
| Moonraker variant payload mismatch (Neptune mods) | Medium | Medium | Compatibility shim + normalized internal enums |
| Unsafe manual Z reset to wrong baseline | Medium | Medium | `/reset_z_offset [target_mm]` baseline-default + explicit zero requirement + verification |
| Emergency cancel command failures | High | Low | Dedicated bypass client, bounded retries, `EMERGENCY_FAILED` degraded mode and urgent notifications |
| First-layer false positives due to lighting/contrast | Medium | Medium | Calibration + ROI guard + bounded actuation + per-filament opt-out |
| Bambu local interface instability | Medium | Medium | Adapter isolation, reconnect/backoff, non-blocking fallback notifications |
| Moonraker WebSocket disconnection during monitoring | High | Medium | Exponential backoff reconnection, full state poll on reconnect, state reconciliation |
| Unauthorized Telegram bot access controlling hardware | High | Low | User/chat ID allowlist, reject unauthorized messages |

---

## 8. Open Questions

- [ ] Should OrcaSlicer become co-equal with PrusaSlicer in MVP, or remain optional profile-compatible backend?
- [ ] Should external prints support user-approved first-layer correction in a future mode, or remain permanently NOTHING-only?
- [ ] What is the minimum integration-test matrix for legacy Moonraker variants (stock, Elegoo-modified, and one pinned older release)?
- [ ] For Bambu, should cloud API be supported as fallback when local MQTT is unavailable, or explicitly local-only for safety/privacy?
