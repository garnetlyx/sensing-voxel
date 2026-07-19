/**
 * Sensing Voxel Dashboard - WebSocket client and DOM updates
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
  const AUTH_STORAGE_KEY = "sensingVoxelApiToken";
  const WS_APPLICATION_PROTOCOL = "sensing-voxel.v1";
  const WS_AUTH_PROTOCOL_PREFIX = "sensing-voxel.auth.";
  const nativeFetch = window.fetch.bind(window);
  let apiToken = sessionStorage.getItem(AUTH_STORAGE_KEY);
  let dashboardStarted = false;
  let cameraAbortController = null;
  let cameraFrameUrl = null;

  function encodeWebSocketToken(token) {
    const bytes = new TextEncoder().encode(token);
    let binary = "";
    bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
    return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
  }

  function setAuthError(message) {
    const error = document.querySelector("#auth-error");
    if (error) error.textContent = message || "";
  }

  function showAuthentication(message = "") {
    document.body.classList.add("auth-required");
    setAuthError(message);
    const input = document.querySelector("#auth-token");
    if (input) {
      input.value = "";
      input.focus();
    }
  }

  function clearAuthentication(message = "Your session is not authorized. Enter the token again.") {
    sessionStorage.removeItem(AUTH_STORAGE_KEY);
    apiToken = null;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (ws) {
      ws.onclose = null;
      ws.close();
      ws = null;
    }
    if (cameraAbortController) {
      cameraAbortController.abort();
      cameraAbortController = null;
    }
    showAuthentication(message);
  }

  async function authenticatedFetch(input, init = {}) {
    if (!apiToken) throw new Error("Dashboard authentication is required");
    const headers = new Headers(init.headers || {});
    headers.set("Authorization", `Bearer ${apiToken}`);
    const response = await nativeFetch(input, { ...init, headers });
    if (response.status === 401) clearAuthentication();
    return response;
  }

  async function authenticate(token) {
    apiToken = token;
    const response = await authenticatedFetch("/api/v1/status");
    if (!response.ok) {
      if (response.status !== 401) apiToken = null;
      throw new Error(response.status === 401 ? "Invalid API token" : "Unable to verify API token");
    }
    sessionStorage.setItem(AUTH_STORAGE_KEY, token);
    document.body.classList.remove("auth-required");
    setAuthError("");
    startDashboard();
  }

  const state = {
    agentState: "UNKNOWN",
    activePrinter: null,
    selectedPrinter: null,
    printers: {},
    job: null,
    alerts: [],
    decisionToken: null,
    decisionJobId: null,
    selectedFile: null,
  };

  const jobHistoryState = {
    jobs: [],
    page: 1,
    perPage: 20,
    total: 0,
    sortField: "started_at",
    sortDesc: true,
    filterStatus: "",
    filterPrinter: "",
    loading: false,
  };

  const queueState = {
    items: [],
    loading: false,
  };

  // --- DOM refs ---
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // --- WebSocket ---
  function connectWebSocket() {
    if (!apiToken) return;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/api/v1/ws`;
    const authProtocol = WS_AUTH_PROTOCOL_PREFIX + encodeWebSocketToken(apiToken);

    ws = new WebSocket(url, [WS_APPLICATION_PROTOCOL, authProtocol]);

    ws.onopen = () => {
      if (ws.protocol !== WS_APPLICATION_PROTOCOL) {
        ws.close(1002, "Required application protocol was not negotiated");
        clearAuthentication("The server rejected dashboard authentication.");
        return;
      }
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
    if (reconnectTimer || !apiToken || document.body.classList.contains("auth-required")) return;
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

    if (data.printers) {
      state.printers = {};
      for (const p of data.printers) {
        state.printers[p.id] = {
          connected: p.connected || false,
          telemetry: p.telemetry || null,
        };
      }
    }

    if (!state.selectedPrinter && Object.keys(state.printers).length > 0) {
      state.selectedPrinter = state.activePrinter || Object.keys(state.printers)[0];
    }

    updatePrinterSelect();
    updatePrinterFilters();
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

  function createElement(tag, className = "", text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  }

  function replaceWithMessage(container, tag, className, text) {
    container.replaceChildren(createElement(tag, className, text));
  }

  // --- Printer selector ---
  function updatePrinterSelect() {
    const select = $("#printer-select");
    if (!select) return;

    const printerIds = Object.keys(state.printers);
    select.replaceChildren();
    if (printerIds.length === 0) {
      const option = createElement("option", "", "No printers");
      option.value = "";
      select.appendChild(option);
      return;
    }

    printerIds.forEach((pid) => {
      const printer = state.printers[pid];
      const dot = printer.connected ? "\u25cf" : "\u25cb";
      const active = pid === state.activePrinter ? " *" : "";
      const option = createElement("option", "", `${dot} ${pid}${active}`);
      option.value = pid;
      option.selected = pid === state.selectedPrinter;
      select.appendChild(option);
    });
  }

  function updatePrinterFilters() {
    const filter = $("#job-printer-filter");
    if (!filter) return;

    const printerIds = Object.keys(state.printers);
    const currentVal = filter.value;
    filter.replaceChildren();
    const allOption = createElement("option", "", "All Printers");
    allOption.value = "";
    filter.appendChild(allOption);
    printerIds.forEach((pid) => {
      const option = createElement("option", "", pid);
      option.value = pid;
      filter.appendChild(option);
    });

    if (currentVal && printerIds.includes(currentVal)) {
      filter.value = currentVal;
    }
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
    const dot = $("#connection-dot");
    const chatDot = $("#chat-connection-dot");
    const label = $(".connection-label");

    if (chatDot) chatDot.classList.toggle("connected", serverConnected);

    const printerConnected = state.selectedPrinter &&
      state.printers[state.selectedPrinter] &&
      state.printers[state.selectedPrinter].connected;

    const bothConnected = serverConnected && printerConnected;
    if (dot) dot.classList.toggle("connected", bothConnected);

    if (label) {
      if (bothConnected) {
        label.textContent = "Connected";
      } else if (serverConnected && printerConnected === false) {
        label.textContent = "Printer Offline";
      } else {
        label.textContent = "Disconnected";
      }
    }
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
    const content = panel.querySelector(".job-content");
    if (!content) return;

    const job = state.job;
    if (!job) {
      replaceWithMessage(content, "p", "no-job", "No active job");
      return;
    }

    const progressNumber = Number(job.progress);
    const progress = Number.isFinite(progressNumber)
      ? Math.max(0, Math.min(100, progressNumber * 100))
      : 0;
    const progressText = progress.toFixed(1);
    const layers = job.current_layer != null && job.total_layers != null
      ? `${job.current_layer} / ${job.total_layers}`
      : "--";
    const time = job.print_time != null ? formatTime(job.print_time) : "--";

    const filename = createElement("div", "job-filename", job.job_file || job.filename || "Unknown");
    const progressContainer = createElement("div", "progress-container");
    const progressBar = createElement("div", "progress-bar");
    const progressFill = createElement("div", "progress-fill");
    progressFill.style.width = `${progress}%`;
    progressBar.appendChild(progressFill);
    const labels = createElement("div", "progress-text");
    labels.append(createElement("span", "", `${progressText}%`), createElement("span", "", time));
    progressContainer.append(progressBar, labels);
    const metadata = createElement("div", "job-meta");
    metadata.append(
      createElement("span", "", `Job: ${job.job_id || "--"}`),
      createElement("span", "", `Layers: ${layers}`)
    );
    content.replaceChildren(filename, progressContainer, metadata);
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
    if (nozzleTarget) nozzleTarget.textContent = ` / ${(temps.extruder_target || 0).toFixed(0)}`;
    if (nozzleBar) {
      const pct = temps.extruder_target > 0
        ? Math.min(100, (temps.extruder / temps.extruder_target) * 100)
        : 0;
      nozzleBar.style.width = pct + "%";
    }

    if (bed) bed.textContent = (temps.bed || 0).toFixed(1);
    if (bedTarget) bedTarget.textContent = ` / ${(temps.bed_target || 0).toFixed(0)}`;
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
      replaceWithMessage(content, "p", "no-decision", "No pending decisions");
      return;
    }

    const isApproval = state.agentState === "AWAITING_APPROVAL";
    const isPaused = state.agentState === "PAUSED_PENDING_USER";
    const prompt = createElement(
      "p",
      "decision-prompt",
      isApproval ? "Print job awaiting your approval." : "Print paused. What would you like to do?"
    );
    const buttons = createElement("div", "decision-buttons");
    const addDecisionButton = (label, className, action) => {
      const button = createElement("button", `btn ${className}`, label);
      button.type = "button";
      button.addEventListener("click", () => window.sendDecision(action));
      buttons.appendChild(button);
    };
    if (isApproval) {
      addDecisionButton("Approve", "btn-approve", "approve");
      addDecisionButton("Reject", "btn-reject", "reject");
    } else if (isPaused) {
      addDecisionButton("Resume", "btn-resume", "resume");
      addDecisionButton("Cancel Print", "btn-cancel", "cancel");
    }
    content.replaceChildren(prompt, buttons);
  }

  // --- Alert management ---
  function chatRoleClass(role) {
    return ["assistant", "system", "user"].includes(role) ? role : "assistant";
  }

  function alertTypeClass(type) {
    return ["error", "info", "success", "warning"].includes(type) ? type : "info";
  }

  function addAlert(type, message) {
    const alert = {
      type: alertTypeClass(type),
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
      replaceWithMessage(list, "p", "no-alerts", "No recent alerts");
      return;
    }

    const fragment = document.createDocumentFragment();
    state.alerts.forEach((alert) => {
      const item = createElement("div", `alert-item ${alert.type}`);
      item.append(
        createElement("div", `alert-dot ${alert.type}`),
        createElement("div", "alert-content", alert.message),
        createElement("div", "alert-time", formatAlertTime(alert.timestamp))
      );
      fragment.appendChild(item);
    });
    list.replaceChildren(fragment);
  }

  // --- Decision submission ---
  window.sendDecision = async function (action) {
    if (!state.decisionToken) return;

    $$(".decision-buttons .btn").forEach((b) => (b.disabled = true));

    try {
      const resp = await authenticatedFetch("/api/v1/decision", {
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

  function formatDateTime(iso) {
    if (!iso) return "--";
    try {
      const d = new Date(iso);
      return d.toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "--";
    }
  }

  function formatDuration(seconds) {
    if (!seconds) return "--";
    return formatTime(seconds);
  }

  async function startCameraStream() {
    const image = $("#camera-feed");
    const noFeed = $("#camera-no-feed");
    if (!image || !apiToken || cameraAbortController) return;

    cameraAbortController = new AbortController();
    try {
      const response = await authenticatedFetch("/api/v1/camera/stream", {
        signal: cameraAbortController.signal,
      });
      if (!response.ok || !response.body) {
        if (noFeed) noFeed.style.display = "flex";
        return;
      }

      const reader = response.body.getReader();
      let bytes = new Uint8Array();
      while (apiToken) {
        const { value, done } = await reader.read();
        if (done) break;
        const combined = new Uint8Array(bytes.length + value.length);
        combined.set(bytes);
        combined.set(value, bytes.length);
        bytes = combined;

        let start = -1;
        let end = -1;
        for (let index = 0; index < bytes.length - 1; index++) {
          if (start < 0 && bytes[index] === 0xff && bytes[index + 1] === 0xd8) {
            start = index;
          } else if (start >= 0 && bytes[index] === 0xff && bytes[index + 1] === 0xd9) {
            end = index + 2;
            break;
          }
        }
        if (start >= 0 && end > start) {
          const blob = new Blob([bytes.slice(start, end)], { type: "image/jpeg" });
          const nextUrl = URL.createObjectURL(blob);
          image.src = nextUrl;
          image.style.display = "block";
          if (noFeed) noFeed.style.display = "none";
          if (cameraFrameUrl) URL.revokeObjectURL(cameraFrameUrl);
          cameraFrameUrl = nextUrl;
          bytes = bytes.slice(end);
        } else if (start > 0) {
          bytes = bytes.slice(start);
        }
      }
    } catch (error) {
      if (error.name !== "AbortError" && noFeed) noFeed.style.display = "flex";
    } finally {
      cameraAbortController = null;
    }
  }

  // --- Camera PTZ ---
  window.cameraMove = async function (direction) {
    try {
      const resp = await authenticatedFetch("/api/v1/camera/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ direction }),
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        addAlert("error", `Camera move failed: ${data.error || resp.statusText}`);
      }
    } catch (e) {
      addAlert("error", `Camera move failed: ${e.message}`);
    }
  };

  // --- AI Chat ---
  let chatPending = false;

  window.sendChatMessage = async function () {
    if (chatPending) return;

    const input = $("#chat-input");
    let message = input.value.trim();

    if (state.selectedFile && !message) {
      message = `I've uploaded ${state.selectedFile.name}. What can you do with it?`;
    }

    if (!message && !state.selectedFile) return;

    chatPending = true;
    updateChatSendButton();
    input.value = "";

    let uploadedFileInfo = null;
    if (state.selectedFile) {
      const fileToUpload = state.selectedFile;
      updateFileIndicator(null);
      appendChatMessage("system", `Uploading ${fileToUpload.name}...`);

      uploadedFileInfo = await uploadFile(fileToUpload);
      if (!uploadedFileInfo) {
        chatPending = false;
        updateChatSendButton();
        return;
      }

      appendChatMessage("system", `Uploaded ${fileToUpload.name} successfully.`);

      const systemHint = `\n\n[System: User uploaded file '${uploadedFileInfo.filename}' to server storage. Use upload_to_printer to send it to the printer if requested.]`;

      appendChatMessage("user", message);

      message += systemHint;
    } else {
      appendChatMessage("user", message);
    }

    const typingEl = appendChatMessage("assistant", "Thinking...");
    typingEl.classList.add("typing");

    try {
      const resp = await authenticatedFetch("/api/v1/chat", {
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
      replaceWithMessage(
        messages,
        "div",
        "chat-welcome",
        "Ask me anything about 3D printing or your printer status."
      );
    }
    try {
      await authenticatedFetch("/api/v1/chat/clear", { method: "POST" });
    } catch {
    }
  };

  function appendChatMessage(role, text) {
    const messages = $("#chat-messages");
    if (!messages) return;

    const welcome = messages.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const el = document.createElement("div");
    el.className = `chat-msg ${chatRoleClass(role)}`;
    el.textContent = text;
    messages.appendChild(el);
    scrollChatToBottom();
    return el;
  }

  function scrollChatToBottom() {
    const messages = $("#chat-messages");
    if (messages) messages.scrollTop = messages.scrollHeight;
  }

  // --- File Upload Helpers ---
  function initFileUpload() {
    const messages = $("#chat-messages");
    const fileInput = $("#chat-file-input");

    if (!messages || !fileInput) return;

    ["dragenter", "dragover"].forEach(name => {
      messages.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        messages.classList.add("drag-over");
      }, false);
    });

    ["dragleave", "drop"].forEach(name => {
      messages.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        messages.classList.remove("drag-over");
      }, false);
    });

    messages.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files[0]) {
        handleFileSelect(files[0]);
      }
    }, false);

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["gcode", "bgcode"].includes(ext)) {
      addAlert("warning", "Only .gcode or .bgcode files are supported currently.");
    }

    state.selectedFile = file;
    updateFileIndicator(file.name);
  }

  function updateFileIndicator(name) {
    const indicator = $("#file-indicator");
    const fileName = $("#file-name");
    if (indicator && fileName) {
      if (name) {
        fileName.textContent = name;
        indicator.classList.add("active");
      } else {
        indicator.classList.remove("active");
        state.selectedFile = null;
        const fileInput = $("#chat-file-input");
        if (fileInput) fileInput.value = "";
      }
    }
  }

  window.cancelUpload = function () {
    updateFileIndicator(null);
  };

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await authenticatedFetch("/api/v1/upload", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errorText = await resp.text();
        throw new Error(errorText || resp.statusText);
      }
      return await resp.json();
    } catch (e) {
      console.error("Upload failed:", e);
      addAlert("error", `File upload failed: ${e.message}`);
      return null;
    }
  }

  async function loadChatHistory() {
    try {
      const resp = await authenticatedFetch("/api/v1/chat/history");
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.history && data.history.length > 0) {
        data.history.forEach((msg) => {
          let text = "";
          if (typeof msg.content === "string") {
            text = msg.content;
          } else if (Array.isArray(msg.content)) {
            const textBlock = msg.content.find((b) => b.type === "text");
            if (textBlock) text = textBlock.text;
          }

          if (text) {
            const cleanText = text.replace(/\n\n\[System: .*\]/, "");
            if (cleanText.trim()) {
              appendChatMessage(msg.role, cleanText);
            }
          }
        });
      }
    } catch (e) {
      console.error("Failed to load chat history:", e);
    }
  }

  function updateChatSendButton() {
    const btn = $("#chat-send");
    if (btn) btn.disabled = chatPending;
  }

  // --- Job History ---
  async function loadJobHistory() {
    if (jobHistoryState.loading) return;
    jobHistoryState.loading = true;

    try {
      const params = new URLSearchParams({
        page: jobHistoryState.page.toString(),
        per_page: jobHistoryState.perPage.toString(),
      });

      if (jobHistoryState.filterStatus) {
        params.append("status", jobHistoryState.filterStatus);
      }
      if (jobHistoryState.filterPrinter) {
        params.append("printer_id", jobHistoryState.filterPrinter);
      }

      const resp = await authenticatedFetch(`/api/v1/jobs/history?${params}`);
      if (resp.ok) {
        const data = await resp.json();
        jobHistoryState.jobs = data.jobs || [];
        jobHistoryState.total = data.total || 0;
        renderJobHistory();
        updateJobPagination();
      }
    } catch (e) {
      console.error("Failed to load job history:", e);
    } finally {
      jobHistoryState.loading = false;
    }
  }

  function renderJobHistory() {
    const tbody = $("#job-history-body");
    if (!tbody) return;

    if (jobHistoryState.jobs.length === 0) {
      const row = createElement("tr", "no-data");
      const cell = createElement("td", "", "No jobs found");
      cell.colSpan = 6;
      row.appendChild(cell);
      tbody.replaceChildren(row);
      return;
    }

    const fragment = document.createDocumentFragment();
    jobHistoryState.jobs.forEach((job) => {
      const row = createElement("tr");
      const filename = createElement("td", "job-filename-cell", job.file_name || "");
      filename.title = String(job.file_name || "");
      const printer = createElement("td", "", job.printer_id || "");
      const statusCell = createElement("td");
      const safeStatus = ["completed", "failed", "cancelled", "running", "pending"].includes(job.status)
        ? job.status
        : "pending";
      statusCell.appendChild(createElement("span", `job-status-badge ${safeStatus}`, job.status || "unknown"));
      const started = createElement("td", "", formatDateTime(job.started_at));
      const duration = createElement("td", "", formatDuration(job.duration_seconds));
      const filament = Number(job.filament_used_mm);
      const filamentText = Number.isFinite(filament) && filament > 0 ? `${filament.toFixed(1)}mm` : "--";
      row.append(filename, printer, statusCell, started, duration, createElement("td", "", filamentText));
      fragment.appendChild(row);
    });
    tbody.replaceChildren(fragment);
  }

  function updateJobPagination() {
    const pageInfo = $("#job-page-info");
    const prevBtn = $("#job-prev-page");
    const nextBtn = $("#job-next-page");

    if (pageInfo) {
      const totalPages = Math.max(1, Math.ceil(jobHistoryState.total / jobHistoryState.perPage));
      pageInfo.textContent = `Page ${jobHistoryState.page} of ${totalPages}`;
    }

    if (prevBtn) prevBtn.disabled = jobHistoryState.page <= 1;
    if (nextBtn) {
      const totalPages = Math.ceil(jobHistoryState.total / jobHistoryState.perPage);
      nextBtn.disabled = jobHistoryState.page >= totalPages;
    }
  }

  function initJobHistory() {
    const statusFilter = $("#job-status-filter");
    const printerFilter = $("#job-printer-filter");
    const prevBtn = $("#job-prev-page");
    const nextBtn = $("#job-next-page");

    if (statusFilter) {
      statusFilter.addEventListener("change", (e) => {
        jobHistoryState.filterStatus = e.target.value;
        jobHistoryState.page = 1;
        loadJobHistory();
      });
    }

    if (printerFilter) {
      printerFilter.addEventListener("change", (e) => {
        jobHistoryState.filterPrinter = e.target.value;
        jobHistoryState.page = 1;
        loadJobHistory();
      });
    }

    if (prevBtn) {
      prevBtn.addEventListener("click", () => {
        if (jobHistoryState.page > 1) {
          jobHistoryState.page--;
          loadJobHistory();
        }
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener("click", () => {
        const totalPages = Math.ceil(jobHistoryState.total / jobHistoryState.perPage);
        if (jobHistoryState.page < totalPages) {
          jobHistoryState.page++;
          loadJobHistory();
        }
      });
    }

    const table = $("#job-history-table");
    if (table) {
      const headers = table.querySelectorAll("th[data-sort]");
      headers.forEach(th => {
        th.addEventListener("click", () => {
          const field = th.dataset.sort;
          if (jobHistoryState.sortField === field) {
            jobHistoryState.sortDesc = !jobHistoryState.sortDesc;
          } else {
            jobHistoryState.sortField = field;
            jobHistoryState.sortDesc = true;
          }

          headers.forEach(h => {
            h.classList.remove("sort-asc", "sort-desc");
          });
          th.classList.add(jobHistoryState.sortDesc ? "sort-desc" : "sort-asc");

          jobHistoryState.jobs.sort((a, b) => {
            let va = a[field];
            let vb = b[field];

            if (field === "started_at") {
              va = va ? new Date(va).getTime() : 0;
              vb = vb ? new Date(vb).getTime() : 0;
            }

            if (va === null || va === undefined) va = "";
            if (vb === null || vb === undefined) vb = "";

            if (typeof va === "string") va = va.toLowerCase();
            if (typeof vb === "string") vb = vb.toLowerCase();

            if (va < vb) return jobHistoryState.sortDesc ? 1 : -1;
            if (va > vb) return jobHistoryState.sortDesc ? -1 : 1;
            return 0;
          });

          renderJobHistory();
        });
      });
    }

    loadJobHistory();
    setInterval(loadJobHistory, 30000);
  }

  // --- Queue Management ---
  async function loadQueueStatus() {
    if (queueState.loading) return;
    queueState.loading = true;

    try {
      const resp = await authenticatedFetch("/api/v1/queue");
      if (resp.ok) {
        const data = await resp.json();
        renderQueue(data.queues || []);
      }
    } catch (e) {
      console.error("Failed to load queue:", e);
    } finally {
      queueState.loading = false;
    }
  }

  function renderQueue(queues) {
    const container = $("#queue-list");
    if (!container) return;

    const allItems = [];
    queues.forEach(q => {
      if (q.queue_size > 0) {
        allItems.push({
          printer_id: q.printer_id,
          size: q.queue_size,
          processing: q.processing,
        });
      }
    });

    if (allItems.length === 0) {
      replaceWithMessage(container, "p", "no-queue", "Queue is empty");
      return;
    }

    const fragment = document.createDocumentFragment();
    allItems.forEach((item) => {
      const queueItem = createElement("div", "queue-item");
      queueItem.appendChild(createElement("span", "queue-handle", "\u2630"));
      const info = createElement("div", "queue-info");
      info.append(
        createElement("div", "queue-filename", `Queue for ${item.printer_id}`),
        createElement(
          "div",
          "queue-printer",
          `${item.size} pending ${item.processing ? "(processing)" : ""}`.trim()
        )
      );
      queueItem.appendChild(info);
      fragment.appendChild(queueItem);
    });
    container.replaceChildren(fragment);
  }

  window.clearQueue = async function() {
    const selectedPrinter = state.selectedPrinter;
    if (!selectedPrinter) {
      addAlert("warning", "No printer selected");
      return;
    }

    try {
      const resp = await authenticatedFetch("/api/v1/queue/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ printer_id: selectedPrinter }),
      });

      if (resp.ok) {
        addAlert("success", `Queue cleared for ${selectedPrinter}`);
        loadQueueStatus();
      } else {
        const data = await resp.json().catch(() => ({}));
        addAlert("error", data.error || "Failed to clear queue");
      }
    } catch (e) {
      addAlert("error", `Failed to clear queue: ${e.message}`);
    }
  };

  function initQueue() {
    loadQueueStatus();
    setInterval(loadQueueStatus, 10000);
  }

  // --- Polling fallback ---
  async function pollStatus() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    try {
      const resp = await authenticatedFetch("/api/v1/status");
      if (resp.ok) {
        const data = await resp.json();
        state.agentState = data.agent_state || "UNKNOWN";
        updateStatePanel();
      }
    } catch {
    }
  }

  // --- Init ---
  function startDashboard() {
    if (!apiToken) return;
    if (dashboardStarted) {
      connectWebSocket();
      startCameraStream();
      loadChatHistory();
      loadJobHistory();
      loadQueueStatus();
      fetchSvaApprovals();
      return;
    }
    dashboardStarted = true;
    connectWebSocket();
    startCameraStream();
    initFileUpload();
    loadChatHistory();
    initJobHistory();
    initQueue();
    fetchSvaApprovals();

    setInterval(pollStatus, 10000);
    setInterval(fetchSvaApprovals, 15000);

    setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 25000);
  }

  function init() {
    const authForm = $("#auth-form");
    const authInput = $("#auth-token");
    const authSubmit = $("#auth-submit");
    if (authForm && authInput) {
      authForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const token = authInput.value.trim();
        if (!token) return;
        if (authSubmit) authSubmit.disabled = true;
        setAuthError("");
        try {
          await authenticate(token);
        } catch (error) {
          apiToken = null;
          sessionStorage.removeItem(AUTH_STORAGE_KEY);
          showAuthentication(error.message);
        } finally {
          if (authSubmit) authSubmit.disabled = false;
        }
      });
    }

    const logout = $("#auth-logout");
    if (logout) logout.addEventListener("click", () => clearAuthentication("Logged out."));

    const cameraButtons = {
      "#camera-up": "up",
      "#camera-left": "left",
      "#camera-right": "right",
      "#camera-down": "down",
    };
    Object.entries(cameraButtons).forEach(([selector, direction]) => {
      const button = $(selector);
      if (button) button.addEventListener("click", () => window.cameraMove(direction));
    });

    const queueClear = $("#queue-clear");
    if (queueClear) queueClear.addEventListener("click", window.clearQueue);
    const chatClear = $("#chat-clear");
    if (chatClear) chatClear.addEventListener("click", window.clearChat);
    const fileCancel = $("#file-cancel");
    if (fileCancel) fileCancel.addEventListener("click", window.cancelUpload);
    const fileAttach = $("#file-attach");
    const fileInput = $("#chat-file-input");
    if (fileAttach && fileInput) fileAttach.addEventListener("click", () => fileInput.click());
    const chatSend = $("#chat-send");
    if (chatSend) chatSend.addEventListener("click", window.sendChatMessage);

    const select = $("#printer-select");
    if (select) select.addEventListener("change", onPrinterSelectChange);

    const chatInput = $("#chat-input");
    if (chatInput) {
      chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendChatMessage();
        }
      });
    }

    if (apiToken) {
      authenticate(apiToken).catch((error) => clearAuthentication(error.message));
    } else {
      showAuthentication();
    }
  }

  async function fetchSvaApprovals() {
    try {
      const resp = await authenticatedFetch("/api/v1/pending-approvals?limit=10");
      if (!resp.ok) return;
      const data = await resp.json();
      renderSvaApprovals(data.approvals || []);
    } catch (e) { /* silent */ }
  }

  const approvalPreviewUrls = new Set();

  function clearApprovalPreviewUrls() {
    approvalPreviewUrls.forEach((url) => URL.revokeObjectURL(url));
    approvalPreviewUrls.clear();
  }

  function approvalElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function formatApprovalValue(value, suffix = "") {
    if (value === undefined || value === null || value === "") return "Not provided";
    return `${value}${suffix}`;
  }

  function formatApprovalDuration(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return "Not provided";
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.round((seconds % 3600) / 60);
    if (hours && minutes) return `${hours} hr ${minutes} min`;
    if (hours) return `${hours} hr`;
    return `${minutes} min`;
  }

  function addApprovalField(grid, label, value, options = {}) {
    const field = approvalElement("div", `approval-field${options.hash ? " approval-field-hash" : ""}`);
    field.appendChild(approvalElement("span", "approval-field-label", label));
    field.appendChild(approvalElement("span", "approval-field-value", value));
    grid.appendChild(field);
  }

  function addApprovalSection(card, title) {
    const section = approvalElement("section", "approval-section");
    section.appendChild(approvalElement("h4", "approval-section-title", title));
    card.appendChild(section);
    return section;
  }

  function printEvidencePayload(approval) {
    const payload = approval && approval.payload;
    if (!payload || typeof payload !== "object") return null;
    const nestedPayload = payload.payload && typeof payload.payload === "object" ? payload.payload : null;
    const containers = nestedPayload ? [nestedPayload, payload] : [payload];
    let printEvidence = null;
    for (const container of containers) {
      printEvidence = (container.print_request && typeof container.print_request === "object" && container.print_request)
        || (container.print_evidence && typeof container.print_evidence === "object" && container.print_evidence)
        || (container.evidence && typeof container.evidence === "object" && container.evidence)
        || (container.request && typeof container.request === "object" && container.request);
      if (printEvidence) break;
    }
    printEvidence = printEvidence || nestedPayload || payload;
    const evidence = Object.assign({}, payload, nestedPayload || {}, printEvidence);
    const confirmation = [
      printEvidence.confirmation,
      nestedPayload && nestedPayload.confirmation,
      payload.confirmation,
    ].find((value) => value && typeof value === "object") || {};
    const binding = [
      printEvidence.binding,
      nestedPayload && nestedPayload.binding,
      payload.binding,
    ].find((value) => value && typeof value === "object") || {};
    const renderedPreview = evidence.rendered_preview;
    const looksLikePrint = approval.tool_name === "start_print_with_monitor"
      || renderedPreview
      || evidence.artifact_sha256
      || evidence.preview_sha256
      || evidence.rendered_preview_sha256
      || evidence.validation_sha256
      || evidence.confirmation_sha256;
    return looksLikePrint ? { evidence, confirmation, binding, renderedPreview } : null;
  }


  function renderPrintEvidence(card, printData) {
    const { evidence, confirmation, binding, renderedPreview } = printData;
    const setupSection = addApprovalSection(card, "Physical print setup");
    const setupGrid = approvalElement("div", "approval-evidence-grid");
    addApprovalField(setupGrid, "Material", formatApprovalValue(confirmation.material));
    addApprovalField(setupGrid, "Build plate", formatApprovalValue(confirmation.build_plate_surface));
    addApprovalField(setupGrid, "Nozzle size", formatApprovalValue(confirmation.hotend_nozzle_size_mm, " mm"));
    addApprovalField(setupGrid, "Probe Z offset", formatApprovalValue(binding.probe_z_offset, " mm"));
    addApprovalField(setupGrid, "Homing-origin Z", formatApprovalValue(binding.homing_origin_z, " mm"));
    addApprovalField(setupGrid, "Confirmation Z", formatApprovalValue(confirmation.z_offset_mm, " mm"));
    addApprovalField(setupGrid, "Estimated time", formatApprovalDuration(confirmation.estimated_print_time_seconds));
    addApprovalField(setupGrid, "Filament usage", formatApprovalValue(confirmation.filament_usage_mm, " mm"));
    setupSection.appendChild(setupGrid);

    const hashesSection = addApprovalSection(card, "Validated evidence hashes");
    const hashesGrid = approvalElement("div", "approval-evidence-grid approval-hash-grid");
    addApprovalField(hashesGrid, "Artifact", formatApprovalValue(evidence.artifact_sha256), { hash: true });
    addApprovalField(hashesGrid, "Preview JSON", formatApprovalValue(evidence.preview_sha256), { hash: true });
    addApprovalField(hashesGrid, "Rendered preview", formatApprovalValue(evidence.rendered_preview_sha256), { hash: true });
    addApprovalField(hashesGrid, "Validation", formatApprovalValue(evidence.validation_sha256), { hash: true });
    addApprovalField(hashesGrid, "Confirmation", formatApprovalValue(evidence.confirmation_sha256), { hash: true });
    hashesSection.appendChild(hashesGrid);

    const previewSection = addApprovalSection(card, "Validated toolpath preview");
    if (typeof renderedPreview === "string" && renderedPreview.trim()) {
      const mediaType = evidence.rendered_preview_media_type || "image/svg+xml";
      if (mediaType === "image/svg+xml") {
        const blob = new Blob([renderedPreview], { type: "image/svg+xml" });
        const previewUrl = URL.createObjectURL(blob);
        approvalPreviewUrls.add(previewUrl);
        const image = approvalElement("img", "approval-toolpath-preview");
        image.alt = "Validated G-code toolpath preview";
        image.src = previewUrl;
        image.addEventListener("load", () => {
          URL.revokeObjectURL(previewUrl);
          approvalPreviewUrls.delete(previewUrl);
        }, { once: true });
        image.addEventListener("error", () => {
          URL.revokeObjectURL(previewUrl);
          approvalPreviewUrls.delete(previewUrl);
        }, { once: true });
        previewSection.appendChild(image);
      } else {
        previewSection.appendChild(approvalElement("p", "approval-preview-unavailable", "Preview format is not supported."));
      }
    } else {
      previewSection.appendChild(approvalElement("p", "approval-preview-unavailable", "No rendered preview was provided."));
    }
  }

  function renderGenericPayload(card, payload) {
    if (!payload || typeof payload !== "object") return;
    const section = addApprovalSection(card, "Requested values");
    const grid = approvalElement("div", "approval-evidence-grid");
    Object.entries(payload).forEach(([key, value]) => {
      const display = value && typeof value === "object" ? JSON.stringify(value) : formatApprovalValue(value);
      addApprovalField(grid, key.replaceAll("_", " "), display);
    });
    section.appendChild(grid);
  }

  function renderSvaApprovals(approvals) {
    const container = $("#sva-approvals-list");
    if (!container) return;
    clearApprovalPreviewUrls();
    container.replaceChildren();

    if (!approvals.length) {
      container.appendChild(approvalElement("p", "no-approvals", "No pending approvals"));
      return;
    }

    approvals.forEach((approval) => {
      const card = approvalElement("article", "approval-item");
      card.appendChild(approvalElement("h3", "approval-tool", approval.tool_name || "Unknown tool"));

      const printData = printEvidencePayload(approval);
      if (printData) renderPrintEvidence(card, printData);
      else renderGenericPayload(card, approval.payload);

      if (approval.reason) {
        const reason = addApprovalSection(card, "Why approval is required");
        reason.appendChild(approvalElement("p", "approval-reason", approval.reason));
      }

      const actions = approvalElement("div", "approval-actions");
      const approve = approvalElement("button", "btn btn-approve", "Approve");
      approve.type = "button";
      approve.addEventListener("click", () => window.approveSvaTool(approval.approval_token));
      const reject = approvalElement("button", "btn btn-reject", "Reject");
      reject.type = "button";
      reject.addEventListener("click", () => window.rejectSvaTool(approval.approval_token));
      actions.append(approve, reject);
      card.appendChild(actions);
      container.appendChild(card);
    });
  }

  window.approveSvaTool = async function(token) {
    try {
      const resp = await authenticatedFetch("/api/v1/approve", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({approval_token: token, action: "approve"}),
      });
      const data = await resp.json();
      addAlert(data.status === "approved" ? "success" : "error", `SVA approval: ${data.status || data.error}`);
      fetchSvaApprovals();
    } catch (e) {
      addAlert("error", `Approval failed: ${e.message}`);
    }
  };

  window.rejectSvaTool = async function(token) {
    try {
      const resp = await authenticatedFetch("/api/v1/approve", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({approval_token: token, action: "reject"}),
      });
      const data = await resp.json();
      addAlert("info", `SVA approval rejected`);
      fetchSvaApprovals();
    } catch (e) {
      addAlert("error", `Reject failed: ${e.message}`);
    }
  };

  window.addEventListener("beforeunload", clearApprovalPreviewUrls);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
