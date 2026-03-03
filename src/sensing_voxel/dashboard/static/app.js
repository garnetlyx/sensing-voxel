/**
 * Sensing Voxel Dashboard — WebSocket client and DOM updates
 * Multi-printer aware: events carry printer_id, UI shows selected printer
 */

(function () {
  "use strict";

  // --- State ---
  let ws = null;
  let reconnectTimer = null;
  let reconnectAttempts = 0;
  const MAX_RECONNECT_DELAY = 30000;
  const BASE_RECONNECT_DELAY = 1000;

  const state = {
    agentState: "UNKNOWN",
    activePrinter: null,
    selectedPrinter: null, // currently viewed in UI (dropdown)
    printers: {},          // { pid: { connected, telemetry } }
    job: null,
    alerts: [],
    decisionToken: null,
    decisionJobId: null,
  };

  // --- DOM refs ---
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // --- WebSocket ---
  function connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/api/v1/ws`;

    ws = new WebSocket(url);

    ws.onopen = () => {
      reconnectAttempts = 0;
      updateConnectionDot(true);
      addAlert("info", "Connected to server");
    };

    ws.onclose = () => {
      updateConnectionDot(false);
      scheduleReconnect();
    };

    ws.onerror = () => {
      updateConnectionDot(false);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch (e) {
        console.error("Failed to parse WS message:", e);
      }
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    const delay = Math.min(
      BASE_RECONNECT_DELAY * Math.pow(1.5, reconnectAttempts),
      MAX_RECONNECT_DELAY
    );
    reconnectAttempts++;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectWebSocket();
    }, delay);
  }

  // --- Message handlers ---
  function handleMessage(data) {
    switch (data.type) {
      case "snapshot":
        handleSnapshot(data);
        break;
      case "state_change":
        handleStateChange(data);
        break;
      case "telemetry":
        handleTelemetry(data);
        break;
      case "progress":
        handleProgress(data);
        break;
      case "alert":
        addAlert(data.alert_type, data.message);
        break;
      case "decision_request":
        handleDecisionRequest(data);
        break;
      case "decision_result":
        handleDecisionResult(data);
        break;
      case "pong":
        break;
    }
  }

  function handleSnapshot(data) {
    if (data.state) {
      state.agentState = data.state.agent_state || "UNKNOWN";
    }
    if (data.job) {
      state.job = data.job;
    }
    state.activePrinter = data.active_printer || null;

    // Populate per-printer data
    if (data.printers) {
      state.printers = {};
      for (const p of data.printers) {
        state.printers[p.id] = {
          connected: p.connected || false,
          telemetry: p.telemetry || null,
        };
      }
    }

    // Auto-select first printer if none selected
    if (!state.selectedPrinter && Object.keys(state.printers).length > 0) {
      state.selectedPrinter = state.activePrinter || Object.keys(state.printers)[0];
    }

    updatePrinterSelect();
    updateAll();
    updateConnectionDot(true);
  }

  function handleStateChange(data) {
    state.agentState = data.agent_state || state.agentState;
    updateStatePanel();
    addAlert("info", `State: ${formatState(state.agentState)}`);
  }

  function handleTelemetry(data) {
    const pid = data.printer_id;
    if (pid && state.printers[pid]) {
      state.printers[pid].telemetry = data;
    }
    // Update temps panel if this is the selected printer
    if (pid === state.selectedPrinter) {
      updateTempsPanel();
    }
  }

  function handleProgress(data) {
    if (state.job) {
      if (data.progress !== undefined) state.job.progress = data.progress;
      if (data.filename !== undefined) state.job.filename = data.filename;
      if (data.current_layer !== undefined) state.job.current_layer = data.current_layer;
      if (data.total_layers !== undefined) state.job.total_layers = data.total_layers;
      if (data.print_time !== undefined) state.job.print_time = data.print_time;
    }
    updateJobPanel();
  }

  function handleDecisionRequest(data) {
    state.decisionToken = data.token;
    state.decisionJobId = data.job_id;
    updateDecisionPanel();
    addAlert("warning", `Decision needed for ${data.job_file}`);
  }

  function handleDecisionResult(data) {
    state.decisionToken = null;
    state.decisionJobId = null;
    updateDecisionPanel();
    addAlert("success", `Decision: ${data.action} for ${data.job_id}`);
  }

  // --- Printer selector ---
  function updatePrinterSelect() {
    const select = $("#printer-select");
    if (!select) return;

    const printerIds = Object.keys(state.printers);
    if (printerIds.length === 0) {
      select.innerHTML = '<option value="">No printers</option>';
      return;
    }

    select.innerHTML = printerIds
      .map((pid) => {
        const p = state.printers[pid];
        const dot = p.connected ? "\u25cf" : "\u25cb";
        const active = pid === state.activePrinter ? " *" : "";
        return `<option value="${escapeHtml(pid)}" ${pid === state.selectedPrinter ? "selected" : ""}>${dot} ${escapeHtml(pid)}${active}</option>`;
      })
      .join("");
  }

  function onPrinterSelectChange(e) {
    state.selectedPrinter = e.target.value;
    updateAll();
  }

  // --- UI updates ---
  function updateAll() {
    updateStatePanel();
    updateJobPanel();
    updateTempsPanel();
    updateDecisionPanel();
  }

  function updateConnectionDot(serverConnected) {
    const dot = $(".connection-dot");
    const label = $(".connection-label");
    // Show connection to selected printer, not just server
    const printerConnected = state.selectedPrinter
      && state.printers[state.selectedPrinter]
      && state.printers[state.selectedPrinter].connected;
    const connected = serverConnected && printerConnected;
    if (dot) dot.classList.toggle("connected", connected);
    if (label) label.textContent = connected ? "Connected" : printerConnected === false && serverConnected ? "Printer Offline" : "Disconnected";
  }

  function updateStatePanel() {
    const badge = $(".header-state-badge");
    if (badge) {
      badge.textContent = formatState(state.agentState);
      badge.className = "state-badge header-state-badge " + stateClass(state.agentState);
    }

    const stateValue = $("#agent-state");
    if (stateValue) stateValue.textContent = formatState(state.agentState);

    const printerConn = $("#printer-connection");
    if (printerConn) {
      const sel = state.selectedPrinter;
      const connected = sel && state.printers[sel] && state.printers[sel].connected;
      printerConn.textContent = connected ? "Connected" : "Disconnected";
      printerConn.style.color = connected ? "var(--success)" : "var(--danger)";
    }
  }

  function updateJobPanel() {
    const panel = $(".job-panel");
    if (!panel) return;

    const job = state.job;
    if (!job) {
      panel.querySelector(".job-content").innerHTML =
        '<p class="no-job">No active job</p>';
      return;
    }

    const progress = job.progress != null ? (job.progress * 100).toFixed(1) : 0;
    const layers = job.current_layer != null && job.total_layers != null
      ? `${job.current_layer} / ${job.total_layers}`
      : "--";
    const time = job.print_time != null ? formatTime(job.print_time) : "--";

    panel.querySelector(".job-content").innerHTML = `
      <div class="job-filename">${escapeHtml(job.job_file || job.filename || "Unknown")}</div>
      <div class="progress-container">
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${progress}%"></div>
        </div>
        <div class="progress-text">
          <span>${progress}%</span>
          <span>${time}</span>
        </div>
      </div>
      <div class="job-meta">
        <span>Job: ${escapeHtml(job.job_id || "--")}</span>
        <span>Layers: ${layers}</span>
      </div>
    `;
  }

  function updateTempsPanel() {
    const sel = state.selectedPrinter;
    const tel = sel && state.printers[sel] ? state.printers[sel].telemetry : null;

    const nozzle = $("#nozzle-temp");
    const nozzleTarget = $("#nozzle-target");
    const nozzleBar = $("#nozzle-bar");
    const bed = $("#bed-temp");
    const bedTarget = $("#bed-target");
    const bedBar = $("#bed-bar");

    if (!tel || !tel.temperatures) {
      if (nozzle) nozzle.textContent = "--";
      if (nozzleTarget) nozzleTarget.textContent = "/ --";
      if (nozzleBar) nozzleBar.style.width = "0%";
      if (bed) bed.textContent = "--";
      if (bedTarget) bedTarget.textContent = "/ --";
      if (bedBar) bedBar.style.width = "0%";
      return;
    }

    const temps = tel.temperatures;

    if (nozzle) nozzle.textContent = (temps.extruder || 0).toFixed(1);
    if (nozzleTarget) nozzleTarget.textContent = `/ ${(temps.extruder_target || 0).toFixed(0)}`;
    if (nozzleBar) {
      const pct = temps.extruder_target > 0
        ? Math.min(100, (temps.extruder / temps.extruder_target) * 100)
        : 0;
      nozzleBar.style.width = pct + "%";
    }

    if (bed) bed.textContent = (temps.bed || 0).toFixed(1);
    if (bedTarget) bedTarget.textContent = `/ ${(temps.bed_target || 0).toFixed(0)}`;
    if (bedBar) {
      const pct = temps.bed_target > 0
        ? Math.min(100, (temps.bed / temps.bed_target) * 100)
        : 0;
      bedBar.style.width = pct + "%";
    }
  }

  function updateDecisionPanel() {
    const panel = $(".decision-panel");
    if (!panel) return;

    const content = panel.querySelector(".decision-content");
    if (!state.decisionToken) {
      content.innerHTML = '<p class="no-decision">No pending decisions</p>';
      return;
    }

    const isApproval = state.agentState === "AWAITING_APPROVAL";
    const isPaused = state.agentState === "PAUSED_PENDING_USER";

    let buttons = "";
    if (isApproval) {
      buttons = `
        <button class="btn btn-approve" onclick="sendDecision('approve')">Approve</button>
        <button class="btn btn-reject" onclick="sendDecision('reject')">Reject</button>
      `;
    } else if (isPaused) {
      buttons = `
        <button class="btn btn-resume" onclick="sendDecision('resume')">Resume</button>
        <button class="btn btn-cancel" onclick="sendDecision('cancel')">Cancel Print</button>
      `;
    }

    content.innerHTML = `
      <p class="decision-prompt">
        ${isApproval ? "Print job awaiting your approval." : "Print paused. What would you like to do?"}
      </p>
      <div class="decision-buttons">${buttons}</div>
    `;
  }

  // --- Alert management ---
  function addAlert(type, message) {
    const alert = {
      type: type,
      message: message,
      timestamp: new Date().toISOString(),
    };
    state.alerts.unshift(alert);
    if (state.alerts.length > 50) state.alerts.pop();
    renderAlerts();
  }

  function renderAlerts() {
    const list = $(".alerts-list");
    if (!list) return;

    if (state.alerts.length === 0) {
      list.innerHTML = '<p class="no-alerts">No recent alerts</p>';
      return;
    }

    list.innerHTML = state.alerts
      .map(
        (a) => `
        <div class="alert-item ${a.type}">
          <div class="alert-dot ${a.type}"></div>
          <div class="alert-content">${escapeHtml(a.message)}</div>
          <div class="alert-time">${formatAlertTime(a.timestamp)}</div>
        </div>`
      )
      .join("");
  }

  // --- Decision submission ---
  window.sendDecision = async function (action) {
    if (!state.decisionToken) return;

    // Disable buttons during request
    $$(".decision-buttons .btn").forEach((b) => (b.disabled = true));

    try {
      const resp = await fetch("/api/v1/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: action,
          token: state.decisionToken,
          job_id: state.decisionJobId,
        }),
      });
      const data = await resp.json();
      if (resp.ok) {
        addAlert("success", `Decision sent: ${action}`);
        state.decisionToken = null;
        state.decisionJobId = null;
        updateDecisionPanel();
      } else {
        addAlert("error", `Decision failed: ${data.message || resp.statusText}`);
      }
    } catch (e) {
      addAlert("error", `Decision request failed: ${e.message}`);
    } finally {
      $$(".decision-buttons .btn").forEach((b) => (b.disabled = false));
    }
  };

  // --- Helpers ---
  function formatState(s) {
    if (!s) return "Unknown";
    return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function stateClass(s) {
    if (!s) return "";
    const lower = s.toLowerCase();
    if (lower.includes("idle")) return "idle";
    if (lower.includes("print") || lower.includes("monitor")) return "printing";
    if (lower.includes("pause")) return "paused";
    if (lower.includes("error") || lower.includes("fail")) return "error";
    if (lower.includes("complete")) return "complete";
    if (lower.includes("await")) return "awaiting";
    return "idle";
  }

  function formatTime(seconds) {
    if (!seconds || seconds < 0) return "--";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  function formatAlertTime(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
      return "";
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // --- AI Chat ---
  let chatPending = false;

  window.sendChatMessage = async function () {
    const input = $("#chat-input");
    const message = (input.value || "").trim();
    if (!message || chatPending) return;

    input.value = "";
    appendChatMessage("user", message);
    chatPending = true;
    updateChatSendButton();

    // Show typing indicator
    const typingEl = document.createElement("div");
    typingEl.className = "chat-msg assistant typing";
    typingEl.textContent = "Thinking...";
    $("#chat-messages").appendChild(typingEl);
    scrollChatToBottom();

    try {
      const resp = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      typingEl.remove();

      if (resp.ok) {
        const data = await resp.json();
        appendChatMessage("assistant", data.response || data.error || "No response");
      } else {
        const data = await resp.json().catch(() => ({}));
        appendChatMessage("assistant", data.error || `Error: ${resp.statusText}`);
      }
    } catch (e) {
      typingEl.remove();
      appendChatMessage("assistant", `Connection error: ${e.message}`);
    } finally {
      chatPending = false;
      updateChatSendButton();
    }
  };

  window.clearChat = async function () {
    const messages = $("#chat-messages");
    if (messages) {
      messages.innerHTML = '<div class="chat-welcome">Ask me anything about 3D printing or your printer status.</div>';
    }
    try {
      await fetch("/api/v1/chat/clear", { method: "POST" });
    } catch {
      // Ignore
    }
  };

  function appendChatMessage(role, text) {
    const messages = $("#chat-messages");
    if (!messages) return;

    // Remove welcome message
    const welcome = messages.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const el = document.createElement("div");
    el.className = `chat-msg ${role}`;
    el.textContent = text;
    messages.appendChild(el);
    scrollChatToBottom();
  }

  function scrollChatToBottom() {
    const messages = $("#chat-messages");
    if (messages) messages.scrollTop = messages.scrollHeight;
  }

  function updateChatSendButton() {
    const btn = $("#chat-send");
    if (btn) btn.disabled = chatPending;
  }

  // --- Polling fallback (fetch status every 10s if WS down) ---
  async function pollStatus() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    try {
      const resp = await fetch("/api/v1/status");
      if (resp.ok) {
        const data = await resp.json();
        state.agentState = data.agent_state || "UNKNOWN";
        updateStatePanel();
      }
    } catch {
      // Ignore - WS reconnect will handle it
    }
  }

  // --- Init ---
  function init() {
    // Wire up printer dropdown
    const select = $("#printer-select");
    if (select) select.addEventListener("change", onPrinterSelectChange);

    // Wire up chat input
    const chatInput = $("#chat-input");
    if (chatInput) {
      chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendChatMessage();
        }
      });
    }

    connectWebSocket();
    setInterval(pollStatus, 10000);

    // Ping every 25s to keep WS alive
    setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 25000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
