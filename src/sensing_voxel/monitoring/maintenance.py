"""
Predictive Maintenance Module for Sensing Voxel

Tracks SHAPER_CALIBRATE results over time, detects mechanical degradation
via drift tracking, generates diagnoses, and triggers proactive calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any

from sensing_voxel.storage.database import Database
from sensing_voxel.storage.calibration_repository import CalibrationRepository, CalibrationType

logger = logging.getLogger(__name__)


class ShaperType(Enum):
    """Input shaper types from Klipper"""

    ZV = "zv"
    MZV = "mzv"
    ZVD = "zvd"
    EI = "ei"
    ZVD_EI = "zvd_ei"
    SHAPER = "shaper"
    SMOOTH_SHAPER = "smooth_shaper"
    UNKNOWN = "unknown"


class MaintenanceAlertLevel(Enum):
    """Alert severity levels for maintenance"""

    INFO = auto()  # Informational drift detected
    WARNING = auto()  # Significant drift, monitor closely
    CRITICAL = auto()  # Critical drift, immediate action needed


class MaintenanceStatus(Enum):
    """Overall maintenance status"""

    HEALTHY = auto()  # No issues detected
    DEGRADED = auto()  # Minor drift detected
    NEEDS_ATTENTION = auto()  # Significant issues, calibration recommended
    CRITICAL = auto()  # Critical issues, immediate maintenance required


@dataclass
class ShaperCalibrationResult:
    """SHAPER_CALIBRATE result data"""

    # Resonance frequencies (Hz)
    x_resonance: float
    y_resonance: float

    # Recommended shaper types
    x_shaper: ShaperType
    y_shaper: ShaperType

    # Vibration reduction and smoothing values
    x_vibration_reduction: float | None = None
    y_vibration_reduction: float | None = None
    x_smoothing: float | None = None
    y_smoothing: float | None = None

    # Raw Klipper data for future analysis
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "x_resonance": self.x_resonance,
            "y_resonance": self.y_resonance,
            "x_shaper": self.x_shaper.value,
            "y_shaper": self.y_shaper.value,
            "x_vibration_reduction": self.x_vibration_reduction,
            "y_vibration_reduction": self.y_vibration_reduction,
            "x_smoothing": self.x_smoothing,
            "y_smoothing": self.y_smoothing,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShaperCalibrationResult":
        """Create from dictionary from storage"""
        return cls(
            x_resonance=data["x_resonance"],
            y_resonance=data["y_resonance"],
            x_shaper=ShaperType(data.get("x_shaper", "unknown")),
            y_shaper=ShaperType(data.get("y_shaper", "unknown")),
            x_vibration_reduction=data.get("x_vibration_reduction"),
            y_vibration_reduction=data.get("y_vibration_reduction"),
            x_smoothing=data.get("x_smoothing"),
            y_smoothing=data.get("y_smoothing"),
            raw_data=data.get("raw_data", {}),
        )


@dataclass
class DriftAnalysis:
    """Analysis of drift between two calibration results"""

    axis: str  # "X", "Y", or "combined"
    frequency_drift_hz: float
    frequency_drift_percent: float
    shaper_changed: bool
    previous_shaper: ShaperType | None = None
    current_shaper: ShaperType | None = None
    time_since_last: timedelta | None = None

    @property
    def severity(self) -> MaintenanceAlertLevel:
        """Determine alert severity based on drift percentage"""
        if self.frequency_drift_percent >= 20.0:
            return MaintenanceAlertLevel.CRITICAL
        elif self.frequency_drift_percent >= 10.0:
            return MaintenanceAlertLevel.WARNING
        elif self.frequency_drift_percent >= 5.0:
            return MaintenanceAlertLevel.INFO
        return MaintenanceAlertLevel.INFO

    def generate_diagnosis(self) -> str:
        """Generate human-readable diagnosis"""
        time_str = ""
        if self.time_since_last:
            days = self.time_since_last.days
            if days >= 30:
                time_str = f" since last month"
            elif days >= 7:
                time_str = f" over {days // 7} weeks"
            else:
                time_str = f" over {days} days"

        if self.shaper_changed:
            prev_name = self.previous_shaper.name if self.previous_shaper else "Unknown"
            curr_name = self.current_shaper.name if self.current_shaper else "Unknown"
            return (
                f"{self.axis}-axis shaper changed from {prev_name} "
                f"to {curr_name}{time_str}. "
                f"Resonance shifted {self.frequency_drift_percent:.1f}% "
                f"({self.frequency_drift_hz:.1f} Hz)"
            )
        else:
            return (
                f"{self.axis}-axis resonance shifted {self.frequency_drift_percent:.1f}% "
                f"({self.frequency_drift_hz:.1f} Hz){time_str}"
            )


@dataclass
class MaintenanceDiagnosis:
    """Complete maintenance diagnosis for a printer"""

    printer_id: str
    timestamp: datetime
    status: MaintenanceStatus
    drift_analyses: list[DriftAnalysis]
    recommendations: list[str]
    calibration_overdue: bool
    days_since_calibration: int | None = None

    @property
    def summary(self) -> str:
        """Generate summary message"""
        if self.status == MaintenanceStatus.HEALTHY:
            return f"Printer {self.printer_id} is operating normally"
        elif self.status == MaintenanceStatus.DEGRADED:
            return f"Printer {self.printer_id} showing minor drift"
        elif self.status == MaintenanceStatus.NEEDS_ATTENTION:
            return f"Printer {self.printer_id} requires calibration"
        else:
            return f"Printer {self.printer_id} CRITICAL - immediate maintenance required"

    @property
    def critical_issues(self) -> list[DriftAnalysis]:
        """Get critical drift issues"""
        return [d for d in self.drift_analyses if d.severity == MaintenanceAlertLevel.CRITICAL]

    @property
    def warning_issues(self) -> list[DriftAnalysis]:
        """Get warning drift issues"""
        return [d for d in self.drift_analyses if d.severity == MaintenanceAlertLevel.WARNING]


@dataclass
class PredictiveMaintenanceConfig:
    """Configuration for predictive maintenance"""

    # Drift thresholds (percentage)
    drift_warning_threshold: float = 10.0  # Alert at 10% drift
    drift_critical_threshold: float = 20.0  # Critical at 20% drift

    # Calibration intervals
    calibration_interval_days: int = 30  # Recommended recalibration interval
    calibration_max_age_days: int = 60  # Maximum acceptable age

    # Analysis window
    min_history_days: int = 7  # Minimum history for trend analysis
    trend_analysis_days: int = 90  # Days to look back for trends

    # Alert settings
    enable_proactive_trigger: bool = True
    proactive_trigger_threshold: float = 15.0  # Trigger calibration at 15% drift

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "drift_warning_threshold": self.drift_warning_threshold,
            "drift_critical_threshold": self.drift_critical_threshold,
            "calibration_interval_days": self.calibration_interval_days,
            "calibration_max_age_days": self.calibration_max_age_days,
            "min_history_days": self.min_history_days,
            "trend_analysis_days": self.trend_analysis_days,
            "enable_proactive_trigger": self.enable_proactive_trigger,
            "proactive_trigger_threshold": self.proactive_trigger_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictiveMaintenanceConfig":
        """Create from dictionary"""
        return cls(
            drift_warning_threshold=data.get("drift_warning_threshold", 10.0),
            drift_critical_threshold=data.get("drift_critical_threshold", 20.0),
            calibration_interval_days=data.get("calibration_interval_days", 30),
            calibration_max_age_days=data.get("calibration_max_age_days", 60),
            min_history_days=data.get("min_history_days", 7),
            trend_analysis_days=data.get("trend_analysis_days", 90),
            enable_proactive_trigger=data.get("enable_proactive_trigger", True),
            proactive_trigger_threshold=data.get("proactive_trigger_threshold", 15.0),
        )


class MaintenanceTracker:
    """
    Tracks printer maintenance status and predicts maintenance needs

    Records SHAPER_CALIBRATE results, detects drift, generates diagnoses,
    and triggers proactive calibration when thresholds are exceeded.
    """

    def __init__(
        self,
        database: Database,
        printer_id: str,
        config: PredictiveMaintenanceConfig | None = None,
    ):
        self.db = database
        self.printer_id = printer_id
        self.config = config or PredictiveMaintenanceConfig()
        self._calibration_repo = CalibrationRepository(database)

    def record_shaper_calibration(
        self,
        calibration_id: str,
        result: ShaperCalibrationResult,
        notes: str | None = None,
    ) -> None:
        """
        Record a SHAPER_CALIBRATE result

        Args:
            calibration_id: Unique ID for this calibration record
            result: The shaper calibration result
            notes: Optional notes about this calibration
        """
        values = result.to_dict()

        self._calibration_repo.create(
            calibration_id=calibration_id,
            printer_id=self.printer_id,
            calibration_type=CalibrationType.SHAPER,
            values=values,
            results={
                "diagnoses_generated": [],
                "proactive_triggered": False,
            },
            notes=notes
            or f"Shaper calibration: X={result.x_shaper.name}, Y={result.y_shaper.name}",
        )

        logger.info(
            f"Recorded shaper calibration for {self.printer_id}: "
            f"X={result.x_resonance:.1f}Hz ({result.x_shaper.name}), "
            f"Y={result.y_resonance:.1f}Hz ({result.y_shaper.name})"
        )

    def get_calibration_history(
        self,
        days: int | None = None,
    ) -> list[tuple[datetime, ShaperCalibrationResult]]:
        """
        Get shaper calibration history for this printer

        Args:
            days: Number of days to look back (default: config.trend_analysis_days)

        Returns:
            List of (timestamp, result) tuples, sorted by timestamp (oldest first)
        """
        days = days or self.config.trend_analysis_days

        records = self._calibration_repo.get_calibration_history(
            printer_id=self.printer_id,
            calibration_type=CalibrationType.SHAPER,
            days=days,
        )

        results = []
        for record in records:
            try:
                result = ShaperCalibrationResult.from_dict(record.values)
                results.append((record.performed_at, result))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse calibration record {record.calibration_id}: {e}")
                continue

        # Sort by timestamp (oldest first)
        results.sort(key=lambda x: x[0])
        return results

    def analyze_drift(
        self,
        days: int | None = None,
    ) -> list[DriftAnalysis]:
        """
        Analyze drift between most recent and previous calibration

        Args:
            days: Number of days to look back

        Returns:
            List of drift analyses for X and Y axes
        """
        history = self.get_calibration_history(days=days)

        if len(history) < 2:
            logger.debug(f"Insufficient history for drift analysis: {len(history)} records")
            return []

        # Get most recent and previous calibration
        previous_time, previous = history[-2]
        current_time, current = history[-1]

        time_delta = current_time - previous_time

        analyses = []

        # Analyze X axis
        if previous.x_resonance > 0:
            x_drift_hz = current.x_resonance - previous.x_resonance
            x_drift_percent = abs(x_drift_hz / previous.x_resonance) * 100

            analyses.append(
                DriftAnalysis(
                    axis="X",
                    frequency_drift_hz=x_drift_hz,
                    frequency_drift_percent=x_drift_percent,
                    shaper_changed=current.x_shaper != previous.x_shaper,
                    previous_shaper=previous.x_shaper,
                    current_shaper=current.x_shaper,
                    time_since_last=time_delta,
                )
            )

        # Analyze Y axis
        if previous.y_resonance > 0:
            y_drift_hz = current.y_resonance - previous.y_resonance
            y_drift_percent = abs(y_drift_hz / previous.y_resonance) * 100

            analyses.append(
                DriftAnalysis(
                    axis="Y",
                    frequency_drift_hz=y_drift_hz,
                    frequency_drift_percent=y_drift_percent,
                    shaper_changed=current.y_shaper != previous.y_shaper,
                    previous_shaper=previous.y_shaper,
                    current_shaper=current.y_shaper,
                    time_since_last=time_delta,
                )
            )

        return analyses

    def analyze_trend(
        self,
        days: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Analyze trend over multiple calibrations

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with trend statistics per axis
        """
        history = self.get_calibration_history(days=days)

        if len(history) < 2:
            return {}

        x_freqs = [r.x_resonance for _, r in history]
        y_freqs = [r.y_resonance for _, r in history]

        def calc_stats(values: list[float]) -> dict[str, float]:
            if len(values) < 2:
                return {}

            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance**0.5

            # Calculate trend (simple linear regression slope)
            n = len(values)
            x_vals = list(range(n))
            x_mean = sum(x_vals) / n
            y_mean = mean

            numerator = sum((x_vals[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x - x_mean) ** 2 for x in x_vals)

            slope = numerator / denominator if denominator != 0 else 0

            return {
                "mean": mean,
                "std_dev": std_dev,
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
                "trend_slope": slope,
                "coefficient_of_variation": (std_dev / mean * 100) if mean != 0 else 0,
            }

        return {
            "X": calc_stats(x_freqs),
            "Y": calc_stats(y_freqs),
        }

    def generate_diagnosis(self) -> MaintenanceDiagnosis:
        """
        Generate complete maintenance diagnosis

        Analyzes drift, calibration age, and trends to produce
        actionable recommendations.

        Returns:
            MaintenanceDiagnosis with status and recommendations
        """
        timestamp = datetime.now()

        # Get drift analysis
        drift_analyses = self.analyze_drift()

        # Check calibration age
        latest = self._calibration_repo.get_latest_calibration(
            printer_id=self.printer_id,
            calibration_type=CalibrationType.SHAPER,
        )

        calibration_overdue = False
        days_since_calibration = None

        if latest:
            days_since_calibration = (timestamp - latest.performed_at).days
            calibration_overdue = days_since_calibration > self.config.calibration_max_age_days
        else:
            calibration_overdue = True

        # Determine overall status
        status = MaintenanceStatus.HEALTHY

        if calibration_overdue:
            status = MaintenanceStatus.NEEDS_ATTENTION

        for analysis in drift_analyses:
            if analysis.severity == MaintenanceAlertLevel.CRITICAL:
                status = MaintenanceStatus.CRITICAL
                break
            elif analysis.severity == MaintenanceAlertLevel.WARNING:
                if status != MaintenanceStatus.CRITICAL:
                    status = MaintenanceStatus.NEEDS_ATTENTION
            elif analysis.severity == MaintenanceAlertLevel.INFO:
                if status == MaintenanceStatus.HEALTHY:
                    status = MaintenanceStatus.DEGRADED

        # Generate recommendations
        recommendations = []

        if calibration_overdue:
            if days_since_calibration:
                recommendations.append(
                    f"Calibration overdue by {days_since_calibration - self.config.calibration_interval_days} days. "
                    f"Run SHAPER_CALIBRATE."
                )
            else:
                recommendations.append(
                    "No calibration history found. Run initial SHAPER_CALIBRATE."
                )

        for analysis in drift_analyses:
            if analysis.severity == MaintenanceAlertLevel.CRITICAL:
                recommendations.append(
                    f"CRITICAL: {analysis.generate_diagnosis()}. Immediate calibration required."
                )
            elif analysis.severity == MaintenanceAlertLevel.WARNING:
                recommendations.append(
                    f"WARNING: {analysis.generate_diagnosis()}. Schedule calibration soon."
                )

        # Add trend-based recommendations
        trends = self.analyze_trend()
        for axis, stats in trends.items():
            cv = stats.get("coefficient_of_variation", 0)
            if cv > 10:
                recommendations.append(
                    f"{axis}-axis shows high variability (CV={cv:.1f}%). "
                    f"Consider checking belt tension."
                )

        return MaintenanceDiagnosis(
            printer_id=self.printer_id,
            timestamp=timestamp,
            status=status,
            drift_analyses=drift_analyses,
            recommendations=recommendations,
            calibration_overdue=calibration_overdue,
            days_since_calibration=days_since_calibration,
        )

    def should_trigger_proactive_calibration(self) -> tuple[bool, str | None]:
        """
        Check if proactive calibration should be triggered

        Returns:
            Tuple of (should_trigger, reason)
        """
        if not self.config.enable_proactive_trigger:
            return False, None

        diagnosis = self.generate_diagnosis()

        # Check calibration interval
        if diagnosis.calibration_overdue:
            return True, f"Calibration interval exceeded ({diagnosis.days_since_calibration} days)"

        # Check drift thresholds
        for analysis in diagnosis.drift_analyses:
            if analysis.frequency_drift_percent >= self.config.proactive_trigger_threshold:
                return (
                    True,
                    f"{analysis.axis}-axis drift {analysis.frequency_drift_percent:.1f}% exceeds threshold",
                )

        return False, None

    def get_last_calibration(self) -> tuple[datetime, ShaperCalibrationResult] | None:
        """
        Get the most recent calibration result

        Returns:
            Tuple of (timestamp, result) or None if no history
        """
        latest = self._calibration_repo.get_latest_calibration(
            printer_id=self.printer_id,
            calibration_type=CalibrationType.SHAPER,
        )

        if not latest:
            return None

        try:
            result = ShaperCalibrationResult.from_dict(latest.values)
            return latest.performed_at, result
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse latest calibration: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """
        Get maintenance statistics for this printer

        Returns:
            Dictionary with maintenance statistics
        """
        history = self.get_calibration_history()
        diagnosis = self.generate_diagnosis()
        trends = self.analyze_trend()

        return {
            "printer_id": self.printer_id,
            "total_calibrations": len(history),
            "current_status": diagnosis.status.name,
            "calibration_overdue": diagnosis.calibration_overdue,
            "days_since_calibration": diagnosis.days_since_calibration,
            "active_alerts": len(diagnosis.warning_issues) + len(diagnosis.critical_issues),
            "critical_issues": len(diagnosis.critical_issues),
            "warning_issues": len(diagnosis.warning_issues),
            "trends": trends,
            "config": self.config.to_dict(),
        }


# Simplified API functions for common use cases


def record_shaper_result(
    database: Database,
    printer_id: str,
    axis: str,
    frequency: float,
    recommended_shaper: str,
    calibration_id: str | None = None,
    notes: str | None = None,
) -> None:
    """
    Record a SHAPER_CALIBRATE result for a single axis.

    Args:
        database: Database instance
        printer_id: Printer identifier
        axis: "X" or "Y"
        frequency: Resonance frequency in Hz
        recommended_shaper: Recommended input shaper type (e.g., "zv", "mzv", "ei")
        calibration_id: Optional unique ID (auto-generated if not provided)
        notes: Optional notes
    """
    tracker = MaintenanceTracker(database, printer_id)

    # Build a partial result (single axis)
    shaper_type = ShaperType(recommended_shaper.lower())

    if axis.upper() == "X":
        result = ShaperCalibrationResult(
            x_resonance=frequency,
            y_resonance=0.0,
            x_shaper=shaper_type,
            y_shaper=ShaperType.UNKNOWN,
        )
    elif axis.upper() == "Y":
        result = ShaperCalibrationResult(
            x_resonance=0.0,
            y_resonance=frequency,
            x_shaper=ShaperType.UNKNOWN,
            y_shaper=shaper_type,
        )
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'X' or 'Y'")

    cal_id = calibration_id or f"{printer_id}_shaper_{datetime.now().isoformat()}"
    tracker.record_shaper_calibration(cal_id, result, notes)


def get_baseline(database: Database, printer_id: str, axis: str) -> float | None:
    """
    Get the baseline (first recorded) resonance frequency for an axis.

    Args:
        database: Database instance
        printer_id: Printer identifier
        axis: "X" or "Y"

    Returns:
        Baseline frequency in Hz, or None if no history exists
    """
    axis = axis.upper()
    if axis not in ("X", "Y"):
        raise ValueError(f"Invalid axis: {axis}. Must be 'X' or 'Y'")

    tracker = MaintenanceTracker(database, printer_id)
    history = tracker.get_calibration_history()

    if not history:
        return None

    # Get the oldest (baseline) calibration
    oldest = history[0]
    _, result = oldest

    if axis == "X":
        return result.x_resonance if result.x_resonance > 0 else None
    else:  # Y
        return result.y_resonance if result.y_resonance > 0 else None


def detect_drift(
    database: Database,
    printer_id: str,
    current_frequency: float,
    axis: str,
) -> float | None:
    """
    Detect frequency drift compared to baseline.

    Args:
        database: Database instance
        printer_id: Printer identifier
        current_frequency: Current resonance frequency in Hz
        axis: "X" or "Y"

    Returns:
        Drift percentage (positive = increase, negative = decrease),
        or None if no baseline exists
    """
    axis = axis.upper()
    if axis not in ("X", "Y"):
        raise ValueError(f"Invalid axis: {axis}. Must be 'X' or 'Y'")

    baseline = get_baseline(database, printer_id, axis)
    if baseline is None or baseline == 0:
        return None

    drift = ((current_frequency - baseline) / baseline) * 100
    return drift


def check_calibration_interval(database: Database, printer_id: str) -> int | None:
    """
    Check days since last shaper calibration.

    Args:
        database: Database instance
        printer_id: Printer identifier

    Returns:
        Days since last calibration, or None if no calibration history
    """
    tracker = MaintenanceTracker(database, printer_id)
    latest = tracker.get_last_calibration()

    if not latest:
        return None

    timestamp, _ = latest
    days_since = (datetime.now() - timestamp).days
    return days_since


def should_calibrate(
    database: Database,
    printer_id: str,
    drift_threshold: float = 10.0,
    interval_days: int = 30,
) -> bool:
    """
    Determine if calibration is recommended based on drift or interval.

    Args:
        database: Database instance
        printer_id: Printer identifier
        drift_threshold: Drift percentage threshold (default: 10%)
        interval_days: Maximum days between calibrations (default: 30)

    Returns:
        True if calibration is recommended
    """
    tracker = MaintenanceTracker(database, printer_id)

    # Check calibration interval
    days_since = check_calibration_interval(database, printer_id)
    if days_since is None:
        return True  # No calibration history

    if days_since >= interval_days:
        return True

    # Check drift for both axes
    latest = tracker.get_last_calibration()
    if latest:
        _, result = latest
        for axis, freq in [("X", result.x_resonance), ("Y", result.y_resonance)]:
            if freq > 0:
                drift = detect_drift(database, printer_id, freq, axis)
                if drift is not None and abs(drift) >= drift_threshold:
                    return True

    return False
