"""
Print monitoring script for Sensing Voxel Phase 4.1 Real-World Test.

Monitors print progress via Moonraker API:
- Query printer state and telemetry
- Capture camera snapshots periodically
- Log telemetry to JSONL file
- Detect completion/failure states
- Trigger notifications on state changes

Usage:
    python scripts/monitor_print.py --printer printer.example --duration 3600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sensing_voxel.adapters.klipper import KlipperAdapter
from sensing_voxel.core.models import NormalizedPrinterState, PrintProgress

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRecord:
    """Single telemetry record for logging."""

    timestamp: str
    state: str
    filename: str | None
    progress: float | None
    print_time: float | None
    current_layer: int | None
    total_layers: int | None
    nozzle_temp: float | None
    nozzle_target: float | None
    bed_temp: float | None
    bed_target: float | None
    snapshot_path: str | None = None


class PrintMonitor:
    """Real-time print monitoring with telemetry logging."""

    def __init__(
        self,
        printer_host: str,
        printer_port: int,
        camera_url: str | None,
        log_file: str,
        snapshot_dir: str,
        snapshot_interval: int,
        poll_interval: int,
    ):
        self.printer_host = printer_host
        self.printer_port = printer_port
        self.camera_url = camera_url
        self.log_file = Path(log_file)
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_interval = snapshot_interval
        self.poll_interval = poll_interval

        self.adapter: KlipperAdapter | None = None
        self._last_state: NormalizedPrinterState | None = None
        self._last_snapshot_time: float = 0
        self._snapshot_count: int = 0
        self._start_time: float = 0
        self._running: bool = False

    async def start(self, duration_seconds: int | None = None) -> dict:
        """Start monitoring until completion or timeout."""
        self._running = True
        self._start_time = asyncio.get_event_loop().time()

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.adapter = KlipperAdapter(host=self.printer_host, port=self.printer_port)
        await self.adapter.connect()

        logger.info(f"Started monitoring {self.printer_host}:{self.printer_port}")
        logger.info(f"Telemetry log: {self.log_file}")
        logger.info(f"Snapshots: {self.snapshot_dir}")

        summary = {
            "start_time": datetime.now().isoformat(),
            "printer": f"{self.printer_host}:{self.printer_port}",
            "status": "unknown",
            "records_logged": 0,
            "snapshots_taken": 0,
            "end_time": None,
            "duration_seconds": 0,
        }

        try:
            while self._running:
                loop_start = asyncio.get_event_loop().time()

                progress = await self._poll_printer()
                await self._log_telemetry(progress)
                summary["records_logged"] += 1

                await self._check_snapshot()

                if self._detect_completion(progress):
                    summary["status"] = "completed"
                    logger.info("Print completed!")
                    break

                if self._detect_failure(progress):
                    summary["status"] = "failed"
                    logger.error("Print failed or error detected!")
                    break

                if duration_seconds:
                    elapsed = loop_start - self._start_time
                    if elapsed >= duration_seconds:
                        summary["status"] = "timeout"
                        logger.warning(f"Monitoring timeout after {elapsed:.0f}s")
                        break

                sleep_time = self.poll_interval - (asyncio.get_event_loop().time() - loop_start)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            summary["status"] = "error"
            summary["error"] = str(e)
            logger.error(f"Monitoring error: {e}")
        finally:
            await self.stop()

        summary["end_time"] = datetime.now().isoformat()
        summary["duration_seconds"] = asyncio.get_event_loop().time() - self._start_time
        summary["snapshots_taken"] = self._snapshot_count

        return summary

    async def stop(self) -> None:
        """Stop monitoring and cleanup."""
        self._running = False
        if self.adapter:
            await self.adapter.disconnect()
            self.adapter = None
        logger.info("Monitoring stopped")

    async def _poll_printer(self) -> PrintProgress:
        """Poll printer for current progress."""
        if not self.adapter:
            return PrintProgress(state=NormalizedPrinterState.UNKNOWN)
        return await self.adapter.get_print_progress()

    async def _log_telemetry(self, progress: PrintProgress) -> None:
        """Log telemetry record to JSONL file."""
        record = TelemetryRecord(
            timestamp=datetime.now().isoformat(),
            state=progress.state.name if progress.state else "UNKNOWN",
            filename=progress.filename,
            progress=progress.progress,
            print_time=progress.print_time,
            current_layer=progress.current_layer,
            total_layers=progress.total_layers,
            nozzle_temp=progress.nozzle_temp,
            nozzle_target=progress.nozzle_target,
            bed_temp=progress.bed_temp,
            bed_target=progress.bed_target,
        )

        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")

        if self._last_state != progress.state:
            logger.info(
                f"State: {self._last_state.name if self._last_state else 'None'} -> "
                f"{progress.state.name if progress.state else 'None'} | "
                f"Progress: {progress.progress or 0:.1%} | "
                f"Layer: {progress.current_layer or '?'}/{progress.total_layers or '?'}"
            )
            self._last_state = progress.state

    async def _check_snapshot(self) -> None:
        """Capture camera snapshot if interval elapsed."""
        if not self.camera_url:
            return

        now = asyncio.get_event_loop().time()
        if now - self._last_snapshot_time < self.snapshot_interval:
            return

        await self._capture_snapshot()
        self._last_snapshot_time = now

    async def _capture_snapshot(self) -> None:
        """Capture snapshot from camera."""
        if not self.camera_url:
            return

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.camera_url) as resp:
                    if resp.status == 200:
                        image_data = await resp.read()
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"snapshot_{timestamp}_{self._snapshot_count:04d}.jpg"
                        filepath = self.snapshot_dir / filename

                        with open(filepath, "wb") as f:
                            f.write(image_data)

                        self._snapshot_count += 1
                        logger.debug(f"Snapshot saved: {filepath}")
                    else:
                        logger.warning(f"Snapshot failed: HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"Snapshot error: {e}")

    def _detect_completion(self, progress: PrintProgress) -> bool:
        """Detect if print has completed."""
        if not progress.state:
            return False
        return (
            progress.state
            in (
                NormalizedPrinterState.TERMINAL,
                NormalizedPrinterState.IDLE_NO_ACTIVE_PRINT,
            )
            and self._last_state == NormalizedPrinterState.PRINTING
        )

    def _detect_failure(self, progress: PrintProgress) -> bool:
        """Detect if print has failed."""
        if not progress.state:
            return False
        return progress.state == NormalizedPrinterState.ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor 3D print progress with telemetry logging")
    parser.add_argument(
        "--printer",
        default="printer.example",
        help="Printer hostname or IP (default: printer.example)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=80,
        help="Printer port (default: 80)",
    )
    parser.add_argument(
        "--camera-url",
        help="Camera snapshot URL (e.g., http://camera.local/snapshot.jpg)",
    )
    parser.add_argument(
        "--log-file",
        default="data/print_telemetry.jsonl",
        help="Telemetry log file (default: data/print_telemetry.jsonl)",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="data/snapshots",
        help="Snapshot directory (default: data/snapshots)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=60,
        help="Seconds between snapshots (default: 60)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between printer polls (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Maximum monitoring duration in seconds (optional)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    monitor = PrintMonitor(
        printer_host=args.printer,
        printer_port=args.port,
        camera_url=args.camera_url,
        log_file=args.log_file,
        snapshot_dir=args.snapshot_dir,
        snapshot_interval=args.snapshot_interval,
        poll_interval=args.poll_interval,
    )

    try:
        summary = await monitor.start(duration_seconds=args.duration)

        print("\n" + "=" * 60)
        print("PRINT MONITORING SUMMARY")
        print("=" * 60)
        print(f"Status: {summary['status'].upper()}")
        print(f"Printer: {summary['printer']}")
        print(f"Start: {summary['start_time']}")
        print(f"End: {summary['end_time']}")
        print(f"Duration: {summary['duration_seconds']:.1f}s")
        print(f"Records logged: {summary['records_logged']}")
        print(f"Snapshots taken: {summary['snapshots_taken']}")

        if summary.get("error"):
            print(f"Error: {summary['error']}")

        print("=" * 60)

        return 0 if summary["status"] == "completed" else 1

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


def main():
    try:
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
