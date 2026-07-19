# Sensing Voxel — Product Requirements Document

**Version**: 2.0
**Date**: 2026-07-18
**Status**: Implemented-product baseline

---

## 1. Product Definition

Sensing Voxel is an experimental, locally operated 3D-printer control and monitoring service. Its primary product promise is not autonomous model generation; it is a traceable, fail-closed path from supplied G-code to an authorized printer start, followed by camera-assisted monitoring.

SVA (Sensing-Voxel Agent) exposes that path through Telegram, a web dashboard/API, and LLM-assisted tools. Deterministic local controls remain authoritative for print authorization. Probabilistic camera/LLM analysis is isolated to monitoring and cannot substitute for validation, operator confirmation, or live preflight.

### 1.1 Product Goals

1. Reject unsupported or unsafe G-code before it reaches a printer.
2. Give an operator a deterministic summary and rendered toolpath preview of the exact artifact being approved.
3. Bind approval to the artifact, validation result, target printer, mutable machine/configuration facts, source trust, user identity, and physical evidence derived from the configured printer state.
4. Recheck mutable printer and Z facts immediately before execution and fail closed on drift.
5. Require first-layer monitoring for user-supplied and SVA-generated G-code started through the safety service.
6. Make hardware-affecting actions attributable through explicit decisions and audit records.
7. Keep dashboard and WebSocket access closed unless deployment authentication requirements are met.

### 1.2 Non-Goals

- Certifying a printer, G-code file, or physical environment as safe.
- Replacing firmware limits, thermal protection, fire precautions, supervision, or a hardware emergency stop.
- Automatically correcting first-layer Z offset from camera output.
- Treating LLM judgment as a deterministic safety control.
- Providing a production end-to-end model-slicing or photo-to-print workflow.
- Guaranteeing production support for printer transports other than the configured Klipper/Moonraker path.

### 1.3 Success Criteria

| Area | Required result |
|---|---|
| Whole-file validation | Every executable line is interpreted; unsupported or unsafe semantics reject the artifact |
| Approval integrity | Artifact, validation, previews, physical confirmation, printer, configuration, source, and user bindings are verified before start |
| Live preflight | Any stale or changed bound machine/Z fact blocks execution |
| Monitoring | Required sources cannot complete a safety-managed start without a valid monitor contract and successful monitor start |
| First layer | Required jobs are checked every 60 seconds until first-layer completion |
| Dashboard exposure | Non-loopback binding without an API token is rejected at startup |
| HTTP authentication | Only `/health` is public; absent or invalid bearer credentials fail closed elsewhere |
| WebSocket access | Token and origin policy are enforced during connection establishment |
| Decision races | A decision is consumed at most once; duplicate side effects are prevented |

---

## 2. Users and Supported Scenarios

### 2.1 Primary User

A technically capable operator running one or more locally reachable Klipper/Moonraker printers who wants remote intake, review, controlled starts, status, and camera-assisted monitoring without surrendering final approval to an LLM.

### 2.2 Supported Runtime Scenarios

#### UC-01: Authorize and Start Supplied G-code

1. An authenticated user submits G-code and selects a printer.
2. The runtime retains the exact bytes in a content-addressed artifact store and computes a SHA-256 identity.
3. The validator semantically interprets every executable line in order while maintaining machine state.
4. Unsupported commands/parameters or unsafe motion, extrusion, temperature, or state transitions produce a rejection.
5. The runtime generates deterministic JSON and rendered SVG previews bound to the artifact and validation result.
6. The user reviews the result and explicitly approves the request, including its material/print metadata and physical snapshot derived from the configured build-plate memo, hotend/nozzle configuration, and live Z-offset state.
7. `PrintSafetyService` verifies the approval binding and performs a first live preflight against current printer and Z facts.
8. For `user_gcode` and `sva_generated`, the service requires a complete monitor contract, validates camera/LLM/printer readiness, and starts `PrintMonitor`.
9. The service performs a second live preflight and then starts the exact printer-side artifact through the target adapter.

**Required result**: Only the exact, validated, approved artifact starts on the exact printer under the bound conditions.

#### UC-02: Review Deterministic Previews

The approval interface exposes:

- a deterministic machine-readable summary;
- a deterministic SVG rendering of the extrusion toolpath;
- hashes for the source artifact, validation result, summary, and rendered preview;
- validation findings, envelopes, and first-layer information available from the interpreted file.

The rendered preview is bounded to 20,000 extrusion segments and 2,000,000 SVG bytes.

**Required result**: Repeated generation from the same validated input yields the same approval artifacts; changed input invalidates the prior binding.

#### UC-03: Confirm Physical Readiness

Before approval can authorize execution, the request includes material and print metadata plus physical evidence derived from the configured build-plate memo, hotend/nozzle configuration, and live Z-offset state. The operator explicitly reviews and approves this bound snapshot.

**Required result**: Missing, incomplete, expired, or mismatched confirmation prevents authorization.

**Boundary**: The snapshot combines configured and live evidence for human review; it is not machine vision or sensor proof of bed clearance, correct material, intact hardware, or safe surroundings.

#### UC-04: Live Preflight

Immediately before printer side effects, the service compares approved bindings with current values for the selected printer, including the available hardware/profile/config/macro digests, probe Z offset, homing-origin Z, source trust, and user identity.

**Required result**: A mismatch, unavailable required fact, disconnected printer, or adapter failure stops execution without guessing.

#### UC-05: Mandatory First-Layer Monitoring

For a safety-managed print whose source is `user_gcode` or `sva_generated`:

1. Validation must provide the information needed to construct the monitor contract.
2. The contract must include filename, source identity, and first-layer Z.
3. `PrintMonitor` captures a camera image and requests independent LLM image analysis at a fixed 60-second cadence until first-layer completion.
4. After first-layer completion, it uses the configured normal check interval.
5. An unacknowledged detected anomaly pauses the printer and produces a notification/alert through configured runtime channels.

**Required result**: The print is not reported as successfully started through the safety service if mandatory monitoring cannot start.

**Boundary**: Monitoring observes and may pause; it does not modify Z offset or guarantee detection within less than 60 seconds.

#### UC-06: Dashboard and API Access

1. `/health` remains accessible for health checks.
2. Every other HTTP path requires the configured bearer credential and fails closed if authentication is unavailable or invalid.
3. Binding the dashboard to a host other than `127.0.0.1`, `::1`, or `localhost` requires an API token.
4. WebSocket setup applies token and allowed-origin checks.
5. Authenticated clients may use available status, telemetry, chat, approval, upload, audit, task, alert, camera, and WebSocket functions.

**Required result**: The application does not silently expose printer controls because a token was omitted.

#### UC-07: Telegram Operation

Telegram commands and uploads require the configured user allowlist. Decisions flow through the same idempotent consumption rules used by other interfaces.

**Required result**: Unauthorized users cannot invoke printer operations, and concurrent callback/timeout paths cannot execute an approval twice.

---

## 3. Functional Requirements

### 3.1 Unified Print Safety Service

- The runtime shall create one `PrintSafetyService` for each configured printer adapter.
- All safety-managed print starts shall pass through this service as the final authorization gate.
- The service shall use immutable content identity rather than trusting a mutable filesystem path.
- Authorization shall fail closed on missing services, adapters, bindings, confirmations, or monitor prerequisites.
- Adapter upload/start failures shall not be converted into successful authorization receipts.

### 3.2 Whole-File Semantic Validation

The validator shall:

- parse every executable line rather than inspect only a prefix or sample;
- maintain relevant machine state across lines;
- enforce supported command and parameter allowlists;
- validate motion bounds and semantics using interpreted coordinates and modes;
- validate extrusion and thermal semantics supported by the safety model;
- identify toolpath and first-layer envelopes and first-layer Z where derivable;
- produce structured issues and a deterministic result digest;
- reject unknown safety-relevant semantics instead of assuming they are harmless.

Validation is specific to the supported dialect and safety model. It is not firmware emulation, physical collision simulation, or proof that a mechanically valid file will print successfully.

### 3.3 Preview and Approval Binding

- Summary JSON and rendered SVG bytes shall be deterministic for a given validated artifact.
- Their digests shall be bound to the source artifact and validation result.
- Approval shall identify the target printer and approving user.
- Approval shall include the material/print metadata and physical snapshot derived from the configured build-plate memo, hotend/nozzle configuration, and live Z-offset state.
- Any changed bound value shall require renewed validation and/or approval as appropriate.

### 3.4 Printer and Z Preflight

- Preflight shall occur after approval verification, again after mandatory monitor startup, and before printer start effects.
- It shall re-read all required mutable facts available through the adapter/runtime.
- Probe Z offset and homing-origin Z shall be compared with the approved snapshot under configured policy/tolerances.
- Preflight shall reject target-printer changes, identity changes, configuration drift, trust changes, and unavailable required data.

### 3.5 Monitoring

- `MANDATORY_MONITOR_SOURCES` shall include `user_gcode` and `sva_generated`.
- A required `MonitorStartContract` shall contain filename, source identity, and first-layer Z.
- Required first-layer checks shall occur every 60 seconds until first-layer completion.
- Normal monitoring cadence shall be configurable for subsequent checks.
- Required monitor startup shall validate printer progress, camera availability and view, and independent LLM analysis readiness before the printer starts.
- Image analysis shall run in an independent context so chat history does not bias the observation.
- The monitor shall treat image/model output as probabilistic evidence and use its configured pause/acknowledgement behavior.

### 3.6 Interfaces and Security

- Telegram shall enforce its configured user allowlist for commands and uploads.
- Dashboard middleware shall leave only `/health` public.
- Protected HTTP requests shall require valid bearer authentication.
- Dashboard startup shall reject non-loopback binding without an API token.
- WebSocket handshakes shall enforce configured authentication and origin policy.
- Deployment documentation shall not imply that application authentication supplies TLS, firewalling, VPN, or network segmentation.

### 3.7 Auditing and Persistence

- Decisions and LLM tool operations shall retain enough identity and outcome information for operational audit.
- Agent state shall use the implemented persistence mechanism and idempotent decision controls.
- Printer memos may store physical facts that do not belong in printer firmware configuration.
- Safety artifacts kept solely in the in-memory content-addressed store are process-lifetime data and shall not be presented as durable history.

---

## 4. Technical Constraints

- **Runtime**: Python 3.12 and `asyncio`.
- **Primary printer path**: Klipper/Moonraker REST and WebSocket APIs.
- **Safety authority**: Deterministic local validation, binding checks, and preflight; LLM output is not authorization authority.
- **Artifact identity**: SHA-256 content identity for source and derived approval artifacts.
- **Monitoring dependency**: Camera access and a configured image-capable LLM are operational dependencies for required monitoring.
- **Preview limits**: At most 20,000 rendered extrusion segments and 2,000,000 SVG bytes.
- **First-layer interval**: Fixed at 60 seconds for required first-layer checks.
- **API exposure**: A token is mandatory for non-loopback dashboard binding.

---

## 5. Risks and Mitigations

| Risk | Impact | Implemented mitigation | Residual limitation |
|---|---|---|---|
| Malicious or unsupported G-code | Hardware damage or unsafe motion | Whole-file stateful validation and default rejection | Validator models only supported semantics and cannot simulate all physical effects |
| Artifact changes after review | Wrong bytes reach printer | Content-addressed storage and digest-bound approval | In-memory artifact availability ends at process restart |
| Printer/configuration drift after approval | Approval no longer describes execution context | Live identity/configuration/Z preflight | Only queryable and modeled facts can be compared |
| Operator misses a physical hazard | Collision, fire, damage | Required physical-confirmation snapshot | Assertion is not sensor verification |
| Camera/LLM misses a defect | Print continues incorrectly | Mandatory cadence and independent image analysis | Occlusion, lighting, latency, outage, and model error remain |
| False anomaly | Unnecessary pause | Acknowledgement/policy handling and operator notification | Probabilistic output cannot eliminate false positives |
| Monitor fails at start | Unmonitored required print | Safety-managed start fails closed | Adapter-side effects may require cleanup/reconciliation after partial failures |
| Exposed dashboard | Unauthorized machine control | Bearer auth, loopback exposure check, WebSocket token/origin enforcement | TLS and network perimeter controls remain deployment responsibilities |
| Concurrent decisions | Duplicate upload/start/cancel effects | Lock-guarded idempotent decision consumption | External systems must still honor returned state and errors |
| Adapter mismatch | Incorrect control behavior | Adapter abstraction and explicit target binding | Klipper/Moonraker is the production-supported path; other adapters need deployment validation |

---

## 6. Actual Limitations

1. Camera monitoring is not deterministic and is not a certified safety system.
2. First-layer monitoring runs every 60 seconds, not continuously and not at a five-second control-loop cadence.
3. No camera-driven first-layer Z-offset actuation is part of the current runtime.
4. The safety service's content-addressed artifact store is in memory.
5. Preview output is intentionally bounded and is not full kinematic, thermal, collision, or material simulation.
6. Physical confirmation depends on attentive review of configured and live evidence and cannot prove every physical condition.
7. The live preflight cannot observe every loose component, obstruction, wiring fault, material mismatch, firmware defect, or environmental risk.
8. A supplied G-code workflow is the complete safety-managed path; model slicing is not an end-to-end production runtime feature.
9. Photo/sketch-to-model generation and automated mesh repair are not current product capabilities.
10. Durable multi-printer scheduling is not part of the current runtime; job-history data is available through the wired repository and API.
11. The Bambu adapter is not production-wired by the default configuration.
12. Dashboard authentication does not itself encrypt traffic or secure the host/network.

---

## 7. Acceptance Tests

The product baseline is acceptable when automated tests demonstrate:

- rejection of unsupported commands, parameters, unsafe state transitions, and out-of-policy motion/thermal behavior anywhere in a file;
- deterministic validation and preview hashes for identical input;
- invalidation on artifact, preview, validation, printer, configuration, Z, source, user, or confirmation mismatch;
- no printer start on failed live preflight;
- no successful safety-managed start for a mandatory source when monitor contract creation or monitor startup fails;
- 60-second monitor scheduling before first-layer completion and configured cadence afterward;
- anomaly-triggered pause behavior under the configured acknowledgement policy;
- only `/health` bypassing HTTP bearer authentication;
- startup rejection for non-loopback dashboard binding without an API token;
- WebSocket rejection for invalid token or disallowed origin;
- one-time decision consumption under concurrent approval/timeout attempts.

Hardware tests remain necessary before relying on a deployment with a real printer.
