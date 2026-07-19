"""
Hardware verification script for Sensing Voxel Phase 4.1 Real-World Test.

Checks connectivity to:
- Configured Klipper printers
- Configured ONVIF camera
- Configured LLM proxy

Output: hardware_status.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import aiohttp

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.camera.tapo import TapoCamera
from sensing_voxel.config import TapoCameraConfig

logger = logging.getLogger(__name__)

DEFAULT_PRINTERS = [
    {"name": "printer_1", "host": "192.0.2.10", "port": 80},
    {"name": "printer_2", "host": "192.0.2.11", "port": 80},
]

DEFAULT_CAMERA_CONFIG = TapoCameraConfig(
    host="192.0.2.20",
    user="camera-user",
    password="camera-password",
    onvif_port=2020,
)

LLM_PROXY_URL = "https://llm.example.com/v1/models"


class HardwareVerifier:
    """Hardware connectivity verifier for Phase 4.1 testing."""

    def __init__(self, output_path: str = "data/hardware_status.json"):
        self.output_path = Path(output_path)
        self.results: dict = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "components": {},
        }

    async def verify_all(self) -> dict:
        """Run all hardware verification checks."""
        logger.info("Starting hardware verification...")

        # Verify printers
        self.results["components"]["printers"] = await self._verify_printers()

        # Verify camera
        self.results["components"]["camera"] = await self._verify_camera()

        # Verify LLM proxy
        self.results["components"]["llm_proxy"] = await self._verify_llm_proxy()

        # Determine overall status
        all_passed = all(
            comp.get("status") == "ok"
            for comp in self.results["components"].values()
            if isinstance(comp, dict)
        )
        self.results["overall_status"] = "ok" if all_passed else "degraded"

        # Save results
        await self._save_results()

        return self.results

    async def _verify_printers(self) -> dict:
        """Verify connectivity to all configured printers."""
        results = {"status": "unknown", "printers": []}

        for printer_cfg in DEFAULT_PRINTERS:
            printer_result = await self._check_printer(printer_cfg)
            results["printers"].append(printer_result)

        # Overall status is ok if at least one printer is reachable
        any_ok = any(p.get("status") == "ok" for p in results["printers"])
        results["status"] = "ok" if any_ok else "failed"

        return results

    async def _check_printer(self, cfg: dict) -> dict:
        """Check connectivity to a single printer."""
        result = {
            "name": cfg["name"],
            "host": cfg["host"],
            "port": cfg["port"],
            "status": "unknown",
            "error": None,
        }

        try:
            adapter = KlipperAdapter(host=cfg["host"], port=cfg["port"])
            await adapter.connect()

            # Quick health check via server info endpoint
            info_result = await adapter._rest_request("GET", "/server/info")

            if info_result.status.value == "success":
                result["status"] = "ok"
                result["klipper_version"] = (
                    info_result.data.get("klipper_version", "unknown")
                    if info_result.data
                    else "unknown"
                )
                result["moonraker_version"] = (
                    info_result.data.get("moonraker_version", "unknown")
                    if info_result.data
                    else "unknown"
                )
                logger.info(f"✓ Printer '{cfg['name']}' at {cfg['host']}:{cfg['port']} - OK")
            else:
                result["status"] = "failed"
                result["error"] = info_result.message
                logger.warning(
                    f"✗ Printer '{cfg['name']}' at {cfg['host']}:{cfg['port']} - {info_result.message}"
                )

            await adapter.disconnect()

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.warning(f"✗ Printer '{cfg['name']}' at {cfg['host']}:{cfg['port']} - {e}")

        return result

    async def _verify_camera(self) -> dict:
        """Verify ONVIF camera connectivity."""
        result = {
            "status": "unknown",
            "host": DEFAULT_CAMERA_CONFIG.host,
            "port": DEFAULT_CAMERA_CONFIG.onvif_port,
            "error": None,
        }

        try:
            camera = TapoCamera(DEFAULT_CAMERA_CONFIG)
            await camera.connect()

            # Try to get presets as a simple connectivity test
            presets = await camera.get_presets()

            if "error" not in presets:
                result["status"] = "ok"
                result["presets_count"] = len(presets.get("presets", []))
                logger.info(
                    f"✓ Camera at {DEFAULT_CAMERA_CONFIG.host}:{DEFAULT_CAMERA_CONFIG.onvif_port} - OK"
                )
            else:
                result["status"] = "failed"
                result["error"] = presets["error"]
                logger.warning(f"✗ Camera - {presets['error']}")

            await camera.disconnect()

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.warning(f"✗ Camera - {e}")

        return result

    async def _verify_llm_proxy(self) -> dict:
        """Verify LLM proxy connectivity."""
        result = {
            "status": "unknown",
            "url": LLM_PROXY_URL,
            "error": None,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(LLM_PROXY_URL) as resp:
                    if resp.status == 200:
                        result["status"] = "ok"
                        result["http_status"] = resp.status
                        logger.info(f"✓ LLM Proxy at {LLM_PROXY_URL} - OK")
                    else:
                        result["status"] = "degraded"
                        result["http_status"] = resp.status
                        result["error"] = f"HTTP {resp.status}"
                        logger.warning(f"⚠ LLM Proxy - HTTP {resp.status}")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.warning(f"✗ LLM Proxy - {e}")

        return result

    async def _save_results(self) -> None:
        """Save verification results to JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Hardware status saved to {self.output_path}")

    def print_summary(self) -> None:
        """Print human-readable summary to console."""
        print("\n" + "=" * 60)
        print("HARDWARE VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Status: {self.results['overall_status'].upper()}")
        print("-" * 60)

        # Printers
        printers = self.results["components"].get("printers", {})
        print("\nPrinters:")
        for printer in printers.get("printers", []):
            status_icon = "✓" if printer["status"] == "ok" else "✗"
            print(f"  {status_icon} {printer['name']} ({printer['host']}:{printer['port']})")
            if printer["status"] == "ok":
                print(f"    Klipper: {printer.get('klipper_version', 'unknown')}")
                print(f"    Moonraker: {printer.get('moonraker_version', 'unknown')}")
            elif printer.get("error"):
                print(f"    Error: {printer['error']}")

        # Camera
        camera = self.results["components"].get("camera", {})
        print("\nCamera:")
        status_icon = "✓" if camera["status"] == "ok" else "✗"
        print(f"  {status_icon} ONVIF Camera ({camera.get('host')}:{camera.get('port')})")
        if camera["status"] == "ok":
            print(f"    Presets: {camera.get('presets_count', 0)}")
        elif camera.get("error"):
            print(f"    Error: {camera['error']}")

        # LLM Proxy
        llm = self.results["components"].get("llm_proxy", {})
        print("\nLLM Proxy:")
        status_icon = "✓" if llm["status"] == "ok" else "⚠" if llm["status"] == "degraded" else "✗"
        print(f"  {status_icon} {llm.get('url')}")
        if llm.get("error"):
            print(f"    Error: {llm['error']}")

        print("\n" + "=" * 60)
        print(f"Full report: {self.output_path}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Verify Sensing Voxel hardware connectivity")
    parser.add_argument(
        "-o",
        "--output",
        default="data/hardware_status.json",
        help="Output JSON file path (default: data/hardware_status.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--camera-host",
        help="Camera IP address (overrides default)",
    )
    parser.add_argument(
        "--camera-user",
        help="Camera username",
    )
    parser.add_argument(
        "--camera-pass",
        help="Camera password",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Override camera config if provided
    global DEFAULT_CAMERA_CONFIG
    if args.camera_host:
        DEFAULT_CAMERA_CONFIG = TapoCameraConfig(
            host=args.camera_host,
            user=args.camera_user or DEFAULT_CAMERA_CONFIG.user,
            password=args.camera_pass or DEFAULT_CAMERA_CONFIG.password,
            onvif_port=DEFAULT_CAMERA_CONFIG.onvif_port,
        )

    # Run verification
    verifier = HardwareVerifier(output_path=args.output)
    asyncio.run(verifier.verify_all())
    verifier.print_summary()

    # Exit with appropriate code
    sys.exit(0 if verifier.results["overall_status"] == "ok" else 1)


if __name__ == "__main__":
    main()
