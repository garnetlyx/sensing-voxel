from __future__ import annotations

import ast
import re
from pathlib import Path

SRC_ROOT = Path(__file__).parents[2] / "src" / "sensing_voxel"
DASHBOARD_STATIC = SRC_ROOT / "dashboard" / "static"
DASHBOARD_JS = DASHBOARD_STATIC / "app.js"
DASHBOARD_HTML = DASHBOARD_STATIC / "index.html"
ALLOWED_CALLERS = {
    Path("adapters/base.py"),
    Path("adapters/bambu.py"),
    Path("adapters/klipper.py"),
    Path("safety/service.py"),
}
ALLOWED_RAW_PRIMITIVE_OWNERS = {
    Path("adapters/klipper.py"),
    Path("infra/gcode.py"),
    Path("safety/parser.py"),
}
RAW_PRINT_START_RE = re.compile(r"(?:^|\b)(?:SDCARD_PRINT_FILE|M23|M24|M32)(?:\b|$)", re.I)


def test_start_print_calls_are_owned_by_adapters_or_safety_service() -> None:
    callers: dict[Path, list[int]] = {}

    for source_path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "start_print"
        ]
        if lines:
            callers[source_path.relative_to(SRC_ROOT)] = lines

    unexpected = {
        path: lines for path, lines in callers.items() if path not in ALLOWED_CALLERS
    }
    assert unexpected == {}, f"start_print bypasses safety ownership: {unexpected}"
    assert Path("safety/service.py") in callers


def test_dashboard_approval_renderer_uses_safe_dom_and_isolated_svg() -> None:
    source = DASHBOARD_JS.read_text(encoding="utf-8")
    renderer_start = source.index("  const approvalPreviewUrls = new Set();")
    renderer_end = source.index("  window.approveSvaTool", renderer_start)
    renderer = source[renderer_start:renderer_end]

    assert "innerHTML" not in renderer
    assert "onclick" not in renderer
    assert "document.createElement" in renderer
    assert ".textContent = String(text)" in renderer
    assert "container.replaceChildren()" in renderer
    assert 'approve.addEventListener("click"' in renderer
    assert 'reject.addEventListener("click"' in renderer
    assert 'new Blob([renderedPreview], { type: "image/svg+xml" })' in renderer
    assert "URL.createObjectURL(blob)" in renderer
    assert "URL.revokeObjectURL(previewUrl)" in renderer
    assert 'approvalElement("img", "approval-toolpath-preview")' in renderer
    assert "DOMParser" not in renderer


def test_dashboard_print_approval_shows_required_safety_evidence() -> None:
    source = DASHBOARD_JS.read_text(encoding="utf-8")
    renderer_start = source.index("  function printEvidencePayload")
    renderer_end = source.index("  function renderGenericPayload", renderer_start)
    renderer = source[renderer_start:renderer_end]

    assert 'approval.tool_name === "start_print_with_monitor"' in renderer
    for wrapper in ("print_request", "print_evidence", "evidence", "request"):
        assert f"container.{wrapper}" in renderer

    expected_fields = {
        "Material": "confirmation.material",
        "Build plate": "confirmation.build_plate_surface",
        "Nozzle size": "confirmation.hotend_nozzle_size_mm",
        "Probe Z offset": "binding.probe_z_offset",
        "Homing-origin Z": "binding.homing_origin_z",
        "Confirmation Z": "confirmation.z_offset_mm",
        "Estimated time": "confirmation.estimated_print_time_seconds",
        "Filament usage": "confirmation.filament_usage_mm",
        "Artifact": "evidence.artifact_sha256",
        "Preview JSON": "evidence.preview_sha256",
        "Rendered preview": "evidence.rendered_preview_sha256",
        "Validation": "evidence.validation_sha256",
        "Confirmation": "evidence.confirmation_sha256",
    }
    for label, expression in expected_fields.items():
        assert f'"{label}"' in renderer
        assert expression in renderer


def test_dashboard_non_print_approvals_remain_text_only_and_readable() -> None:
    source = DASHBOARD_JS.read_text(encoding="utf-8")
    renderer_start = source.index("  function renderGenericPayload")
    renderer_end = source.index("  window.approveSvaTool", renderer_start)
    renderer = source[renderer_start:renderer_end]

    assert "Object.entries(payload)" in renderer
    assert "JSON.stringify(value)" in renderer
    assert "addApprovalField(grid" in renderer
    assert "innerHTML" not in renderer


def test_dashboard_browser_auth_transport_is_secret_safe() -> None:
    source = DASHBOARD_JS.read_text(encoding="utf-8")

    assert 'sessionStorage.getItem(AUTH_STORAGE_KEY)' in source
    assert 'sessionStorage.setItem(AUTH_STORAGE_KEY, token)' in source
    assert 'sessionStorage.removeItem(AUTH_STORAGE_KEY)' in source
    assert "localStorage" not in source
    assert 'headers.set("Authorization", `Bearer ${apiToken}`)' in source
    assert 'if (response.status === 401) clearAuthentication();' in source
    assert 'new WebSocket(url, [WS_APPLICATION_PROTOCOL, authProtocol])' in source
    assert 'const url = `${proto}//${location.host}/api/v1/ws`;' in source
    assert "apiToken}`" not in source[source.index("function connectWebSocket"):source.index("function scheduleReconnect")]


def test_dashboard_dynamic_rendering_avoids_html_and_secret_attributes() -> None:
    source = DASHBOARD_JS.read_text(encoding="utf-8")
    html = DASHBOARD_HTML.read_text(encoding="utf-8")

    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in source
    assert re.search(r"\.dataset\.[A-Za-z_$][\w$]*\s*=", source) is None
    assert "approval_token" not in html
    assert re.search(r"\son[a-z]+\s*=", html, re.IGNORECASE) is None
    assert '<img id="camera-feed" class="camera-feed"\n             alt="Camera feed">' in html
    assert "document.createElement" in source
    assert ".textContent" in source
    assert ".replaceChildren(" in source
    assert 'option.value = pid;' in source
    assert 'filename.title = String(job.file_name || "");' in source
    assert 'image.src = previewUrl;' in source
    assert 'addEventListener("click"' in source


def test_dashboard_server_derived_css_classes_are_allowlisted() -> None:
    source = DASHBOARD_JS.read_text(encoding="utf-8")

    assert 'return ["assistant", "system", "user"].includes(role) ? role : "assistant";' in source
    assert 'return ["error", "info", "success", "warning"].includes(type) ? type : "info";' in source
    assert '["completed", "failed", "cancelled", "running", "pending"].includes(job.status)' in source
    assert 'el.className = `chat-msg ${chatRoleClass(role)}`;' in source


def test_raw_print_start_primitives_are_owned_by_policy_or_adapter() -> None:
    owners: dict[Path, list[int]] = {}

    for source_path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and RAW_PRINT_START_RE.search(node.value)
        ]
        if lines:
            owners[source_path.relative_to(SRC_ROOT)] = lines

    unexpected = {
        path: lines for path, lines in owners.items() if path not in ALLOWED_RAW_PRIMITIVE_OWNERS
    }
    assert unexpected == {}, f"raw print-start primitives bypass safety ownership: {unexpected}"
    assert Path("adapters/klipper.py") in owners
    assert Path("infra/gcode.py") in owners
