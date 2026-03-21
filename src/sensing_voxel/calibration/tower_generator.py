"""
Tower G-code generator for temperature and flow calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class TowerType(Enum):
    TEMPERATURE = auto()
    FLOW = auto()


@dataclass
class TempTowerConfig:
    """Configuration for temperature tower."""

    start_temp: int = 220
    end_temp: int = 180
    step_temp: int = 5
    base_height: float = 2.0
    section_height: float = 7.0
    tower_size: float = 20.0
    filament_diameter: float = 1.75
    nozzle_diameter: float = 0.4
    layer_height: float = 0.2
    print_speed: int = 3000
    travel_speed: int = 6000
    fan_speed: int = 255
    bed_temp: int = 60

    def __post_init__(self):
        if self.start_temp <= self.end_temp:
            raise ValueError("start_temp must be greater than end_temp")
        if self.step_temp <= 0:
            raise ValueError("step_temp must be positive")
        if (self.start_temp - self.end_temp) % self.step_temp != 0:
            raise ValueError("temperature range must be divisible by step")

    @property
    def temperatures(self) -> list[int]:
        temps = []
        temp = self.start_temp
        while temp >= self.end_temp:
            temps.append(temp)
            temp -= self.step_temp
        return temps

    @property
    def num_sections(self) -> int:
        return len(self.temperatures)

    @property
    def total_height(self) -> float:
        return self.base_height + (self.num_sections * self.section_height)


@dataclass
class FlowTowerConfig:
    """Configuration for flow tower."""

    start_flow: int = 90
    end_flow: int = 110
    step_flow: int = 5
    base_height: float = 2.0
    section_height: float = 7.0
    tower_size: float = 20.0
    temp: int = 200
    filament_diameter: float = 1.75
    nozzle_diameter: float = 0.4
    layer_height: float = 0.2
    print_speed: int = 3000
    travel_speed: int = 6000
    fan_speed: int = 255
    bed_temp: int = 60

    def __post_init__(self):
        if self.start_flow >= self.end_flow:
            raise ValueError("start_flow must be less than end_flow")
        if self.step_flow <= 0:
            raise ValueError("step_flow must be positive")
        if (self.end_flow - self.start_flow) % self.step_flow != 0:
            raise ValueError("flow range must be divisible by step")

    @property
    def flows(self) -> list[int]:
        flows = []
        flow = self.start_flow
        while flow <= self.end_flow:
            flows.append(flow)
            flow += self.step_flow
        return flows

    @property
    def num_sections(self) -> int:
        return len(self.flows)

    @property
    def total_height(self) -> float:
        return self.base_height + (self.num_sections * self.section_height)


@dataclass
class TowerSection:
    """A single section of a calibration tower."""

    index: int
    value: int  # Temperature or flow percentage
    start_z: float
    end_z: float
    center_x: float
    center_y: float
    size: float

    @property
    def height(self) -> float:
        return self.end_z - self.start_z

    @property
    def mid_z(self) -> float:
        return (self.start_z + self.end_z) / 2


@dataclass
class GeneratedTower:
    """Result of tower G-code generation."""

    gcode: str
    tower_type: TowerType
    sections: list[TowerSection]
    config: TempTowerConfig | FlowTowerConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.gcode, encoding="utf-8")


class TowerGenerator:
    """Generate G-code for temperature and flow calibration towers."""

    DEFAULT_START_GCODE = """G28
G1 Z5 F3000
M104 S{temp}
M140 S{bed_temp}
M190 S{bed_temp}
M109 S{temp}
G1 Z0.28 F240
G92 E0
G1 Y140 E10 F1500
G1 X2.3 F5000
G1 Y10 E20 F1200
G92 E0
"""

    DEFAULT_END_GCODE = """G91
G1 Z+0.5 F3000
G90
G1 X10 Y220 F6000
M104 S0
M140 S0
M107
M84
"""

    def __init__(
        self,
        start_gcode: str | None = None,
        end_gcode: str | None = None,
    ):
        self.start_gcode = start_gcode or self.DEFAULT_START_GCODE
        self.end_gcode = end_gcode or self.DEFAULT_END_GCODE

    def generate_temp_tower(
        self,
        config: TempTowerConfig | None = None,
    ) -> GeneratedTower:
        config = config or TempTowerConfig()
        sections: list[TowerSection] = []
        gcode_lines: list[str] = []

        gcode_lines.append("; Temperature Tower Calibration")
        gcode_lines.append(f"; Range: {config.start_temp}C to {config.end_temp}C")
        gcode_lines.append(f"; Step: {config.step_temp}C")
        gcode_lines.append("")

        start_temp = config.temperatures[0] if config.temperatures else 200
        gcode_lines.append(self.start_gcode.format(temp=start_temp, bed_temp=config.bed_temp))

        center_x, center_y = config.tower_size / 2 + 20, config.tower_size / 2 + 20

        self._generate_base(gcode_lines, config, center_x, center_y, start_temp)

        current_z = config.base_height
        for i, temp in enumerate(config.temperatures):
            section = TowerSection(
                index=i,
                value=temp,
                start_z=current_z,
                end_z=current_z + config.section_height,
                center_x=center_x,
                center_y=center_y,
                size=config.tower_size,
            )
            sections.append(section)

            self._generate_section(gcode_lines, section, config, temp, is_temp=True)
            current_z += config.section_height

        gcode_lines.append(self.end_gcode)

        metadata = {
            "type": "temperature",
            "start_temp": config.start_temp,
            "end_temp": config.end_temp,
            "step_temp": config.step_temp,
            "total_sections": len(sections),
            "total_height": config.total_height,
        }

        return GeneratedTower(
            gcode="\n".join(gcode_lines),
            tower_type=TowerType.TEMPERATURE,
            sections=sections,
            config=config,
            metadata=metadata,
        )

    def generate_flow_tower(
        self,
        config: FlowTowerConfig | None = None,
    ) -> GeneratedTower:
        config = config or FlowTowerConfig()
        sections: list[TowerSection] = []
        gcode_lines: list[str] = []

        gcode_lines.append("; Flow Tower Calibration")
        gcode_lines.append(f"; Range: {config.start_flow}% to {config.end_flow}%")
        gcode_lines.append(f"; Step: {config.step_flow}%")
        gcode_lines.append("")

        gcode_lines.append(self.start_gcode.format(temp=config.temp, bed_temp=config.bed_temp))

        center_x, center_y = config.tower_size / 2 + 20, config.tower_size / 2 + 20

        self._generate_base(gcode_lines, config, center_x, center_y, config.temp)

        current_z = config.base_height
        for i, flow in enumerate(config.flows):
            section = TowerSection(
                index=i,
                value=flow,
                start_z=current_z,
                end_z=current_z + config.section_height,
                center_x=center_x,
                center_y=center_y,
                size=config.tower_size,
            )
            sections.append(section)

            self._generate_section(gcode_lines, section, config, flow, is_temp=False)
            current_z += config.section_height

        gcode_lines.append(self.end_gcode)

        metadata = {
            "type": "flow",
            "start_flow": config.start_flow,
            "end_flow": config.end_flow,
            "step_flow": config.step_flow,
            "total_sections": len(sections),
            "total_height": config.total_height,
        }

        return GeneratedTower(
            gcode="\n".join(gcode_lines),
            tower_type=TowerType.FLOW,
            sections=sections,
            config=config,
            metadata=metadata,
        )

    def _generate_base(
        self,
        gcode_lines: list[str],
        config: TempTowerConfig | FlowTowerConfig,
        center_x: float,
        center_y: float,
        temp: int,
    ) -> None:
        half = config.tower_size / 2
        layers = int(config.base_height / config.layer_height)

        gcode_lines.append(f"; Base section ({config.base_height}mm)")
        gcode_lines.append(f"M104 S{temp}")
        gcode_lines.append("")

        for layer in range(layers):
            z = (layer + 1) * config.layer_height
            gcode_lines.append(f"; Layer {layer + 1}, Z={z:.2f}")
            gcode_lines.append(f"G1 Z{z:.2f} F{config.travel_speed}")

            gcode_lines.append(self._rect_perimeter_gcode(center_x, center_y, half, config))

        gcode_lines.append("")

    def _generate_section(
        self,
        gcode_lines: list[str],
        section: TowerSection,
        config: TempTowerConfig | FlowTowerConfig,
        value: int,
        is_temp: bool,
    ) -> None:
        half = section.size / 2
        layers = int(section.height / config.layer_height)

        if is_temp:
            gcode_lines.append(f"; Section {section.index + 1}: {value}C")
            gcode_lines.append(f"M104 S{value} ; Set temperature")
        else:
            gcode_lines.append(f"; Section {section.index + 1}: {value}% flow")
            gcode_lines.append(f"M221 S{value} ; Set flow percentage")

        gcode_lines.append(f"; Z range: {section.start_z:.2f} - {section.end_z:.2f}")
        gcode_lines.append("")

        for layer in range(layers):
            z = section.start_z + ((layer + 1) * config.layer_height)
            gcode_lines.append(f"; Layer {layer + 1}, Z={z:.2f}")
            gcode_lines.append(f"G1 Z{z:.2f} F{config.travel_speed}")

            gcode_lines.append(
                self._rect_perimeter_gcode(section.center_x, section.center_y, half, config)
            )

            gcode_lines.append(
                self._text_indicator_gcode(section.center_x, section.center_y, z, value, config)
            )

        gcode_lines.append("")

    def _rect_perimeter_gcode(
        self,
        cx: float,
        cy: float,
        half: float,
        config: TempTowerConfig | FlowTowerConfig,
    ) -> str:
        extrusion_per_mm = self._calculate_extrusion(config)
        perimeter = 8 * half
        e_total = perimeter * extrusion_per_mm * 2

        lines = [
            "G92 E0",
            f"G1 X{cx - half} Y{cy - half} F{config.travel_speed}",
            f"G1 X{cx + half} Y{cy - half} E{e_total * 0.25:.4f} F{config.print_speed}",
            f"G1 X{cx + half} Y{cy + half} E{e_total * 0.5:.4f}",
            f"G1 X{cx - half} Y{cy + half} E{e_total * 0.75:.4f}",
            f"G1 X{cx - half} Y{cy - half} E{e_total:.4f}",
        ]
        return "\n".join(lines)

    def _text_indicator_gcode(
        self,
        cx: float,
        cy: float,
        z: float,
        value: int,
        config: TempTowerConfig | FlowTowerConfig,
    ) -> str:
        return f"M117 Zone: {value}"

    def _calculate_extrusion(
        self,
        config: TempTowerConfig | FlowTowerConfig,
    ) -> float:
        nozzle_area = math.pi * (config.nozzle_diameter / 2) ** 2
        filament_area = math.pi * (config.filament_diameter / 2) ** 2
        return (nozzle_area * config.layer_height) / filament_area
