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

  function updatePrinterFilters() {
    const filter = $("#job-printer-filter");
    if (!filter) return;

    const printerIds = Object.keys(state.printers);
    const currentVal = filter.value;

    filter.innerHTML = '<option value="">All Printers</option>' +
      printerIds.map(pid => `<option value="${escapeHtml(pid)}">${escapeHtml(pid)}</option>`).join("");

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

  function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // --- Camera PTZ ---
  window.cameraMove = async function (direction) {
    try {
      const resp = await fetch("/api/v1/camera/move", {
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
    }
  };

  function appendChatMessage(role, text) {
    const messages = $("#chat-messages");
    if (!messages) return;

    const welcome = messages.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const el = document.createElement("div");
    el.className = `chat-msg ${role}`;
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
      const resp = await fetch("/api/v1/upload", {
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
      const resp = await fetch("/api/v1/chat/history");
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

      const resp = await fetch(`/api/v1/jobs/history?${params}`);
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
      tbody.innerHTML = '<tr class="no-data"><td colspan="6">No jobs found</td></tr>';
      return;
    }

    tbody.innerHTML = jobHistoryState.jobs.map(job => `
      <tr>
        <td class="job-filename-cell" title="${escapeHtml(job.file_name)}">${escapeHtml(job.file_name)}</td>
        <td>${escapeHtml(job.printer_id)}</td>
        <td><span class="job-status-badge ${job.status}">${job.status}</span></td>
        <td>${formatDateTime(job.started_at)}</td>
        <td>${formatDuration(job.duration_seconds)}</td>
        <td>${job.filament_used_mm ? job.filament_used_mm.toFixed(1) + "mm" : "--"}</td>
      </tr>
    `).join("");
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
      const resp = await fetch("/api/v1/queue");
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
      container.innerHTML = '<p class="no-queue">Queue is empty</p>';
      return;
    }

    container.innerHTML = allItems.map(item => `
      <div class="queue-item">
        <span class="queue-handle">&#9776;</span>
        <div class="queue-info">
          <div class="queue-filename">Queue for ${escapeHtml(item.printer_id)}</div>
          <div class="queue-printer">${item.size} pending ${item.processing ? "(processing)" : ""}</div>
        </div>
      </div>
    `).join("");
  }

  window.clearQueue = async function() {
    const selectedPrinter = state.selectedPrinter;
    if (!selectedPrinter) {
      addAlert("warning", "No printer selected");
      return;
    }

    try {
      const resp = await fetch("/api/v1/queue/clear", {
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
      const resp = await fetch("/api/v1/status");
      if (resp.ok) {
        const data = await resp.json();
        state.agentState = data.agent_state || "UNKNOWN";
        updateStatePanel();
      }
    } catch {
    }
  }

  // --- Init ---
  function init() {
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

    connectWebSocket();
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

  async function fetchSvaApprovals() {
    try {
      const resp = await fetch("/api/v1/pending-approvals?limit=10");
      if (!resp.ok) return;
      const data = await resp.json();
      renderSvaApprovals(data.approvals || []);
    } catch (e) { /* silent */ }
  }

  function renderSvaApprovals(approvals) {
    const container = $("#sva-approvals-list");
    if (!container) return;
    if (!approvals.length) {
      container.innerHTML = '<p class="no-approvals">No pending approvals</p>';
      return;
    }
    container.innerHTML = approvals.map(a => {
      const payloadStr = a.payload ? Object.entries(a.payload).map(([k,v]) => `${k}: ${v}`).join(", ") : "";
      return `<div class="approval-item" data-token="${a.approval_token}">
        <div class="approval-tool">${a.tool_name}</div>
        <div class="approval-payload">${payloadStr}</div>
        <div class="approval-reason">${a.reason || ""}</div>
        <div class="approval-actions">
          <button class="btn btn-approve" onclick="approveSvaTool('${a.approval_token}')">Approve</button>
          <button class="btn btn-reject" onclick="rejectSvaTool('${a.approval_token}')">Reject</button>
        </div>
      </div>`;
    }).join("");
  }

  window.approveSvaTool = async function(token) {
    try {
      const resp = await fetch("/api/v1/approve", {
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
      const resp = await fetch("/api/v1/approve", {
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

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
