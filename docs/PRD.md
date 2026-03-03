# Sensing Voxel - Product Requirements Document

**Version**: 1.1
**Date**: 2026-03-03
**Status**: Approved
**Source**: `.design/plan.md` v10 + `.design/review.md` fixes

---

## 1. Overview

### 1.1 Vision

Sensing Voxel is an end-to-end AI platform that turns photos into physical 3D prints with minimal human intervention. The long-term goal is a single-message workflow: a user sends a photo, and the system generates a 3D model, prepares it for printing, prints it, and monitors the entire process autonomously.

This vision is delivered in phases:
- **Phase 1 (MVP)**: Intelligent print automation — Telegram-driven job intake, automated slicing, camera monitoring (NOTHING detection + first-layer auto-correction), and printer control across a mixed Klipper/Bambu fleet.
- **Phase 2**: Platform maturity — multi-printer orchestration, job history/analytics, full Bambu parity, predictive maintenance.
- **Phase 3**: Photo-to-print pipeline — Image-to-3D generation via AI models, automated mesh repair for printability, and one-tap physical output from any photo or sketch.

### 1.2 Problem Statement

**Near-term**: Current home and prosumer 3D-print workflows are fragmented and manual: users manually slice, tune first layer, watch for NOTHING failures, and inspect calibration towers by eye. This leads to failed overnight prints, wasted filament, and inconsistent quality. The problem is worse in mixed-printer environments where control APIs differ (Moonraker vs Bambu protocols), forcing duplicated manual processes.

**Long-term**: 3D printing remains inaccessible to non-technical users. Even with perfect printer automation, the pipeline breaks at the start: users need CAD skills or must find pre-made models. Image-to-3D generation is advancing rapidly (TripoSR, Meshy, Tripo3D, etc.), but no platform connects AI-generated models to a reliable print pipeline. The generated meshes are not print-ready — they require manifold repair, wall thickening, support analysis, and orientation optimization before slicing. Sensing Voxel aims to close this gap by owning the full chain from image capture to finished print.

### 1.3 Goals & Success Metrics

| Goal | Metric | Target | Phase |
|------|--------|--------|-------|
| Reduce failed prints | Failure rate on monitored jobs | < 5% | 1 |
| Detect defects early | Time from defect onset to pause/cancel | < 30s | 1 |
| Reduce manual workflow | User interactions from file to print start | <= 3 | 1 |
| First-layer quality | First-layer classification accuracy (qualified conditions) | > 80% | 1 |
| Safety race correctness | Duplicate decision actions (approve/resume/timeout races) | 0 in test suite | 1 |
| External print coverage | External active prints detected while agent idle | 100% | 1 |
| Photo-to-print success | Printable mesh generated from single photo | > 70% (decorative objects) | 3 |
| End-to-end autonomy | User interactions from photo to finished print | 1 (send photo + approve) | 3 |

---

## 2. Users & Personas

### 2.1 Target Users

**Phase 1 (MVP)**:
- Hobbyists and makers running Klipper printers (Ender-class, Neptune-class, custom builds).
- Users with Bambu printers who want a unified Telegram-first workflow.
- Small-batch operators managing 2-6 printers and needing unattended monitoring.

**Phase 3 (Photo-to-Print)**:
- Non-technical users who want physical objects from photos without learning CAD or slicing.
- Creators and designers who need rapid physical prototyping from reference images.
- Small businesses wanting on-demand custom prints from product photos.

### 2.2 Persona: Wei - Mixed-Fleet Hobby Operator

- **Background**: Runs an Ender 3 V2 (Klipper/Fluidd), Neptune 4 Max (modified Moonraker/Fluidd), and Bambu P1-series printer.
- **Goals**: Submit models from phone, avoid babysitting, get consistent first layers and fewer failed long prints.
- **Pain Points**: Different printer control surfaces, manual slicer steps, late failure discovery, and repetitive tuning work.

### 2.3 Persona: Mia - Non-Technical Creator (Phase 3)

- **Background**: Runs a small craft shop. Owns a Bambu A1 but rarely uses it because she cannot create 3D models.
- **Goals**: Snap a photo of a product idea, get a physical prototype in hours without learning any software.
- **Pain Points**: CAD tools are intimidating, Thingiverse models never match what she wants, hiring a 3D modeler is slow and expensive.

---

## 3. Use Cases

### UC-01: Geek Customer Service Workflow (Telegram File-to-Print)

**Actor**: Wei
**Trigger**: Wei sends `.stl` or `.step` to Telegram bot.
**Flow**:
1. Agent validates file type/size and stages file.
2. Agent runs headless PrusaSlicer/OrcaSlicer and performs geometry analysis (overhang >60°, bridges, thin walls).
3. Agent **proactively recommends** optimizations based on geometry analysis. Example: detects overhangs >60° and asks user in chat if they want to enable tree supports, with estimated filament cost (e.g., "+15g filament, +20min print time").
4. Agent sends full summary (print time, filament usage, detected issues, recommendations) with inline **Approve/Reject** decision.
5. On approve, agent uploads G-code and starts print.
6. On timeout (default: **10 minutes**, configurable) or reject, agent performs local cleanup and notifies user.

**Success**: Print starts within 2 minutes of approval. Agent acts as a knowledgeable assistant, proactively surfacing issues and suggesting fixes rather than silently slicing.

### UC-02: First-Layer "God Hand" Auto-Correction (Agent-Initiated Jobs Only)

**Actor**: Agent
**Trigger**: Agent-started print enters first-layer monitoring window.
**Flow**:
1. Verify calibration validity (`homography`, `mm_per_pixel`) and `FirstLayerParams` availability.
2. Compute toolpath-aware ROI from first-layer G-code bounds.
3. Evaluate at 5s cadence with 2-consecutive hysteresis and 10s cooldown.
4. Visual classifier analyzes first-layer quality: gaps between lines indicate Z too high, filament scraping/ridging indicates Z too low.
5. Inject bounded Z-offset corrections via Moonraker `SET_GCODE_OFFSET Z_ADJUST` (`±0.01mm` step, clamp `±0.05mm`).
6. **Saturation exit**: if cumulative correction reaches clamp boundary and classifier still reports poor quality for 3 consecutive evaluations, pause print and notify user with current offset and classification result.
7. Restore Z-offset baseline on terminal transition.

**Success**: First layer converges without user intervention and no cross-print drift. The agent acts as a "god hand" — doing what experienced operators do by feel, but via real-time vision.

### UC-03: Multimodal Failure Detection With Controlled Stop

**Actor**: Agent
**Trigger**: Active print monitoring detects sustained failure pattern via vision and/or audio.
**Flow**:
1. **Vision**: NOTHING
2. **Audio** (Phase 2): YAMNet-based audio classifier monitors microphone feed for anomalous sounds — stepper motor skipping, glass bed cracking, filament grinding/clicking at extruder. Audio detection triggers `M112` emergency stop immediately for mechanical safety events.
3. If vision shows high confidence (3/5 frames >0.85), emergency cancel path executes.
4. If vision shows moderate confidence (2/3 frames 0.6-0.85), pause and request **Resume/Cancel**.
5. If pause decision times out (default: **5 minutes**, configurable), auto-cancel active job.

**Success**: Defects are stopped quickly. Audio adds a second sensory channel that catches failures invisible to camera (extruder jams, mechanical failures, bed detachment sounds).

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
**Trigger**: User requests filament calibration via `/calibrate <filament>`.
**Flow**:
1. Agent generates temperature tower and flow-rate tower G-code.
2. Agent prints towers and uses vision to score each zone (surface quality, stringing, layer adhesion).
3. Agent proposes optimal temperature and flow values to user with visual comparison.
4. On approval, agent injects optimal values into Klipper config via Moonraker API and executes `SAVE_CONFIG` to persist across restarts.

**Success**: Filament calibration reduced from multi-hour manual iteration to a single automated cycle. Values are permanently written to printer config, not just session-temporary.
**Scope**: Phase 2.

### UC-06: Predictive Maintenance Tracking

**Actor**: Agent
**Trigger**: Scheduled or user-triggered `SHAPER_CALIBRATE` events.
**Flow**:
1. After each `SHAPER_CALIBRATE` run, agent records X/Y resonance frequency and recommended shaper parameters with timestamp.
2. Agent tracks drift over weeks/months. Significant frequency shift indicates mechanical changes: belt loosening, bearing wear, loose bolts, or frame flex.
3. When drift exceeds threshold, agent notifies user with specific diagnosis (e.g., "X-axis resonance shifted from 42Hz to 36Hz over 3 months — check belt tension and X-axis bearing").
4. Agent can trigger a calibration run proactively if time since last run exceeds configured interval.

**Success**: Mechanical issues (belt wear, bearing degradation) caught before they visibly degrade print quality. Users get actionable maintenance alerts, not just raw numbers.
**Scope**: Phase 2.

### UC-07: Filament Spool Monitoring (Phase 2)

**Actor**: Agent
**Trigger**: Continuous monitoring during active prints via secondary USB camera pointed at spool holder.
**Flow**:
1. Secondary USB camera captures spool images at configurable cadence.
2. Vision estimates remaining spool volume/diameter to predict filament runout before it happens.
3. If remaining filament is below estimated job requirement, agent warns user before print start or mid-print.
4. Vision also detects crossed-filament tangles on the spool — a common cause of mid-print jams that is invisible to filament sensors.
5. On tangle detection, agent pauses print and notifies user to untangle before resuming.

**Success**: Prevents two common failure modes that existing filament runout sensors cannot catch: (a) running out mid-print when remaining quantity is marginal, (b) tangle-induced jams that snap filament or strip the extruder gear.
**Scope**: Phase 2.

### UC-08: Photo-to-Print Pipeline (Phase 3)

**Actor**: Mia
**Trigger**: User sends a photo to Telegram bot.
**Flow**:
1. Agent validates image (resolution, clarity, object isolation).
2. Agent calls Image-to-3D API (e.g., TripoSR, Meshy, Tripo3D) to generate mesh.
3. Agent runs automated mesh repair pipeline via MeshLab/Blender headless scripts: manifold repair, thin-wall thickening, orientation optimization, support analysis. Image-to-3D models (TripoSR, Meshy, Tripo3D) frequently produce non-manifold meshes — the agent acts as a "fixer" to make them print-ready.
4. Agent presents 3D preview (rendered thumbnail) with print estimates (time, filament) and inline **Approve/Reject** decision.
5. On approve, agent enters the existing slicing → upload → print → monitor pipeline (UC-01 flow from step 2 onward).
6. On reject, agent offers adjustment options (scale, orientation, detail level) or cancels.

**Printability gate**: If the mesh repair pipeline cannot produce a structurally sound model (e.g., too thin, non-manifold after repair), the agent reports the issue and does not proceed to print.

**Success**: User receives a physical object matching the photo within hours, with only 1 interaction (send photo) + 1 approval.
**Scope**: Phase 3.

### UC-09: Bambu Printer Interface

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

### 4.2 Phase 2 — Platform Maturity

- [ ] STEP/3MF intake support alongside STL.
- [ ] Audio anomaly detection (YAMNet): stepper skip, glass crack, filament grind/click → `M112` emergency stop (UC-03).
- [ ] Closed-loop calibration tower scoring and `SAVE_CONFIG` injection of optimal temp/flow values (UC-05).
- [ ] Filament spool monitoring: secondary USB camera estimates remaining spool volume to prevent runouts, and visually detects crossed-filament tangles to pause before extruder jam (UC-09).
- [ ] Full Bambu adapter command parity and richer telemetry.
- [ ] SQLite job history and analytics storage.
- [ ] Multi-printer queue scheduling dashboard.
- [ ] Predictive maintenance via `SHAPER_CALIBRATE` drift tracking (UC-06).

### 4.3 Future (Phase 3) — Photo-to-Print

- [ ] Image-to-3D generation via external AI model APIs (TripoSR, Meshy, Tripo3D, or equivalent).
- [ ] Automated mesh repair pipeline: manifold repair, thin-wall thickening, orientation optimization.
- [ ] Printability analysis and gate (reject unprintable meshes before slicing).
- [ ] 3D preview rendering for Telegram (thumbnail of generated model before approval).
- [ ] Multi-view / text-prompt enhancement for higher-quality 3D generation.
- [ ] Sketch-to-print support (hand-drawn sketches as input alongside photos).

### 4.4 Out of Scope

- Training custom Image-to-3D models (use existing APIs/open-source models).
- Direct firmware modification/forking.
- Fully autonomous cloud-only orchestration that bypasses local safety controls.
- Bambu cloud API fallback (local-only for safety/privacy in MVP).
- Functional/mechanical part generation from photos (Phase 3 targets decorative/organic shapes only).

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
| Image-to-3D API quality insufficient for printing | Medium | Medium | Printability gate rejects bad meshes; user shown preview before committing to print |
| Generated mesh structurally unsound (thin walls, non-manifold) | High | High | Automated mesh repair pipeline + conservative rejection threshold |
| Image-to-3D API cost per generation | Medium | Medium | Cache results; let user approve before expensive operations; consider open-source model self-hosting |
| User expectation mismatch (photo vs printed result) | Medium | High | Clear preview with disclaimers; initial scope limited to decorative/organic objects |

---

## 8. Open Questions

### Phase 1 (MVP)
- [ ] Should OrcaSlicer become co-equal with PrusaSlicer in MVP, or remain optional profile-compatible backend?
- [ ] Should external prints support user-approved first-layer correction in a future mode, or remain permanently NOTHING-only?
- [ ] What is the minimum integration-test matrix for legacy Moonraker variants (stock, Elegoo-modified, and one pinned older release)?
- [ ] For Bambu, should cloud API be supported as fallback when local MQTT is unavailable, or explicitly local-only for safety/privacy?

### Phase 3 (Photo-to-Print)
- [ ] Which Image-to-3D backend to prioritize? Cloud API (Meshy, Tripo3D) vs self-hosted open-source (TripoSR, InstantMesh)?
- [ ] What mesh repair library to use? (trimesh, PyMeshLab, or custom pipeline)
- [ ] Should the system support multi-view input (multiple photos of same object) for higher quality generation?
- [ ] How to handle user expectations for functional parts vs decorative objects? Hard gate or soft warning?
