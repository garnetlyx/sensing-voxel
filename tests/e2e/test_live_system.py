"""
E2E tests for live system — verifies dashboard, API, and LLM tool use.
Requires the system to be running on localhost:8080.
"""

import json
import re

import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:8080"


@pytest.fixture(scope="module")
def page(browser):
    """Create a page with extended timeout for LLM calls."""
    page = browser.new_page()
    page.set_default_timeout(60000)  # 60s for LLM responses
    yield page
    page.close()


class TestDashboard:
    """Test web dashboard loads and shows printer data."""

    def test_dashboard_loads(self, page: Page):
        """Dashboard HTML loads successfully."""
        page.goto(BASE_URL)
        expect(page).to_have_title(re.compile(r"Sensing Voxel|Dashboard", re.IGNORECASE))

    def test_printer_status_visible(self, page: Page):
        """Dashboard shows printer connection status."""
        page.goto(BASE_URL)
        # Wait for WebSocket data to populate
        page.wait_for_timeout(3000)
        # Should show at least one printer name
        body_text = page.text_content("body")
        assert "neptune4max" in body_text or "rpi-printer" in body_text, (
            f"No printer name found in dashboard body"
        )


class TestAPIEndpoints:
    """Test REST API endpoints."""

    def test_status_endpoint(self, page: Page):
        """GET /api/v1/status returns valid JSON."""
        resp = page.request.get(f"{BASE_URL}/api/v1/status")
        assert resp.ok
        data = resp.json()
        assert "agent_state" in data
        assert "active_printer" in data
        assert data["printer_connected"] is True

    def test_telemetry_endpoint(self, page: Page):
        """GET /api/v1/telemetry returns printer data."""
        resp = page.request.get(f"{BASE_URL}/api/v1/telemetry")
        assert resp.ok
        data = resp.json()
        # Should have at least one printer's telemetry
        assert isinstance(data, dict)


class TestLLMChat:
    """Test LLM chat with tool use — the core feature."""

    def test_chat_printer_status(self, page: Page):
        """LLM should use get_print_progress tool and return real printer data."""
        resp = page.request.post(
            f"{BASE_URL}/api/v1/chat",
            data=json.dumps({
                "message": "what is the current printing status of my printer?",
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok, f"Chat request failed: {resp.status} {resp.text()}"
        data = resp.json()
        reply = data.get("response", "")
        assert reply, f"Empty reply from chat: {data}"

        # The reply should contain real printer data (temperatures, state, etc.)
        reply_lower = reply.lower()
        has_printer_info = any(keyword in reply_lower for keyword in [
            "print", "temperature", "temp", "nozzle", "bed",
            "idle", "standby", "progress", "petg", "gcode",
            "°c", "celsius", "heating",
            # Chinese responses
            "打印", "温度", "喷嘴", "热床", "空闲", "待机", "进度",
        ])
        assert has_printer_info, (
            f"Reply doesn't seem to contain printer data: {reply[:300]}"
        )
        print(f"\n--- LLM Reply (status query) ---\n{reply}\n---")

    def test_chat_temperature_query(self, page: Page):
        """LLM should report actual temperatures from the printer."""
        resp = page.request.post(
            f"{BASE_URL}/api/v1/chat",
            data=json.dumps({
                "message": "what are the current temperatures?",
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        data = resp.json()
        reply = data.get("response", "")
        assert reply

        # Should contain numeric temperature values
        has_temp = bool(re.search(r"\d+\.?\d*\s*°?[cC]", reply))
        assert has_temp, (
            f"Reply doesn't contain temperature values: {reply[:300]}"
        )
        print(f"\n--- LLM Reply (temp query) ---\n{reply}\n---")

    def test_chat_chinese_response(self, page: Page):
        """LLM should respond in Chinese when asked in Chinese."""
        resp = page.request.post(
            f"{BASE_URL}/api/v1/chat",
            data=json.dumps({
                "message": "打印机现在什么状态？",
            }),
            headers={"Content-Type": "application/json"},
        )
        assert resp.ok
        data = resp.json()
        reply = data.get("response", "")
        assert reply
        print(f"\n--- LLM Reply (Chinese) ---\n{reply}\n---")
