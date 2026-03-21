"""
3D Preview and User Approval for Sensing Voxel.

Provides 3D mesh rendering using matplotlib for preview thumbnails
and multi-angle views. Calculates print estimates and creates
Telegram approval messages with inline keyboards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup


@dataclass
class PrintEstimates:
    """Print time and filament estimates."""

    time_minutes: float
    filament_mm: float
    filament_grams: float | None = None
    layer_height: float | None = None


@dataclass
class ApprovalMessage:
    """Telegram approval message with keyboard."""

    text: str
    keyboard: list[list[InlineKeyboardButton]] | None = None


def render_thumbnail(mesh_path: Path | str, output_path: Path | str) -> Path:
    """
    Render a single thumbnail image from an STL/mesh file using matplotlib 3D.

    Args:
        mesh_path: Path to the input mesh file (STL, OBJ, etc.)
        output_path: Path where the thumbnail image will be saved

    Returns:
        Path to the generated thumbnail image

    Raises:
        FileNotFoundError: If mesh file does not exist
        ValueError: If mesh cannot be loaded or rendered
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    mesh_path = Path(mesh_path)
    output_path = Path(output_path)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    # Load mesh
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh: {mesh_path}")

    # Create figure with dark theme
    fig = plt.figure(figsize=(8, 8), facecolor="#1a1a1a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")

    # Render mesh
    _plot_mesh(ax, mesh, color="#4a90d9")

    # Set viewing angle (isometric-like)
    ax.view_init(elev=30, azim=45)

    # Styling
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Remove grid for cleaner look
    ax.grid(False)
    ax.set_axis_off()

    # Equal aspect ratio
    limits = np.array([mesh.bounds[0], mesh.bounds[1]])
    ax.set_box_aspect([limits[1][i] - limits[0][i] for i in range(3)])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor="#1a1a1a", edgecolor="none")
    plt.close(fig)

    logger.info(f"Thumbnail saved: {output_path}")
    return output_path


def render_multi_angle(
    mesh_path: Path | str,
    output_dir: Path | str,
    views: list[tuple[float, float, str]] | None = None,
) -> list[Path]:
    """
    Generate multi-angle preview images from an STL/mesh file.

    Args:
        mesh_path: Path to the input mesh file
        output_dir: Directory where preview images will be saved
        views: List of (elevation, azimuth, suffix) tuples. Default: front, side, top

    Returns:
        List of paths to generated preview images

    Raises:
        FileNotFoundError: If mesh file does not exist
        ValueError: If mesh cannot be loaded or rendered
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    # Default views: front, side, top
    if views is None:
        views = [
            (0, -90, "front"),  # Front view
            (0, 0, "side"),  # Side view
            (90, -90, "top"),  # Top view
        ]

    # Load mesh
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh: {mesh_path}")

    output_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for elev, azim, suffix in views:
        fig = plt.figure(figsize=(6, 6), facecolor="#1a1a1a")
        ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")

        _plot_mesh(ax, mesh, color="#4a90d9")

        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        # Equal aspect ratio
        limits = np.array([mesh.bounds[0], mesh.bounds[1]])
        ax.set_box_aspect([limits[1][i] - limits[0][i] for i in range(3)])

        output_path = output_dir / f"{mesh_path.stem}_{suffix}.png"
        plt.savefig(output_path, dpi=150, facecolor="#1a1a1a", edgecolor="none")
        plt.close(fig)

        output_paths.append(output_path)
        logger.info(f"View '{suffix}' saved: {output_path}")

    return output_paths


def _plot_mesh(
    ax,
    mesh: trimesh.Trimesh,
    color: str = "#4a90d9",
    edge_color: str = "#2a5a8a",
) -> None:
    """
    Plot a trimesh mesh on a matplotlib 3D axis.

    Args:
        ax: Matplotlib 3D axis
        mesh: Trimesh object to plot
        color: Face color
        edge_color: Edge color
    """
    # Get mesh data
    vertices = mesh.vertices
    faces = mesh.faces

    # Create face colors array
    face_colors = [color] * len(faces)

    # Plot with triangulation
    tri = ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color=color,
        edgecolor=edge_color,
        linewidth=0.1,
        alpha=0.9,
        shade=True,
    )

    # Set equal scaling
    max_range = (
        np.array(
            [
                vertices[:, 0].max() - vertices[:, 0].min(),
                vertices[:, 1].max() - vertices[:, 1].min(),
                vertices[:, 2].max() - vertices[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def calculate_estimates(
    mesh_path: Path | str,
    slicer_config: dict | None = None,
) -> PrintEstimates:
    """
    Calculate print time and filament estimates from a mesh.

    Uses simplified heuristic calculations based on mesh volume,
    surface area, and default slicer settings.

    Args:
        mesh_path: Path to the mesh file
        slicer_config: Optional slicer configuration dict with keys:
            - layer_height: Layer height in mm (default: 0.2)
            - nozzle_diameter: Nozzle diameter in mm (default: 0.4)
            - print_speed: Print speed in mm/s (default: 50)
            - infill_density: Infill percentage 0-1 (default: 0.2)
            - filament_diameter: Filament diameter in mm (default: 1.75)

    Returns:
        PrintEstimates with time_minutes and filament_mm

    Raises:
        FileNotFoundError: If mesh file does not exist
        ValueError: If mesh cannot be loaded
    """
    mesh_path = Path(mesh_path)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    # Load mesh
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh: {mesh_path}")

    # Default slicer settings
    config = {
        "layer_height": 0.2,
        "nozzle_diameter": 0.4,
        "print_speed": 50.0,  # mm/s
        "infill_density": 0.2,
        "filament_diameter": 1.75,
        "wall_line_count": 2,
        "top_bottom_layers": 4,
    }
    if slicer_config:
        config.update(slicer_config)

    # Calculate mesh properties
    volume = mesh.volume if mesh.is_watertight else mesh.bounding_box.volume
    surface_area = mesh.area
    height = mesh.bounds[1][2] - mesh.bounds[0][2]  # Z extent

    # Layer count
    layer_count = max(1, int(height / config["layer_height"]))

    # Extrusion width (typically 110-120% of nozzle diameter)
    extrusion_width = config["nozzle_diameter"] * 1.1

    # Calculate perimeters (walls)
    # Simplified: use surface area to estimate perimeter length
    # Rough approximation: surface_area / (2 * height) gives average perimeter
    avg_perimeter = surface_area / (2 * max(height, config["layer_height"]))
    wall_volume = (
        avg_perimeter
        * config["wall_line_count"]
        * extrusion_width
        * config["layer_height"]
        * layer_count
    )

    # Top/bottom layers
    top_bottom_area = mesh.bounding_box.area / 2  # Rough approximation
    top_bottom_volume = (
        top_bottom_area
        * config["top_bottom_layers"]
        * 2  # Both top and bottom
        * config["layer_height"]
        * extrusion_width
    )

    # Infill volume
    infill_volume = (volume - wall_volume - top_bottom_volume) * config["infill_density"]
    infill_volume = max(0, infill_volume)

    # Total filament volume
    total_volume = wall_volume + top_bottom_volume + infill_volume

    # Convert to filament length
    filament_radius = config["filament_diameter"] / 2
    filament_area = np.pi * filament_radius**2
    filament_mm = total_volume / filament_area

    # Calculate print time
    # Extrusion rate (mm^3/s) = print_speed * layer_height * extrusion_width
    extrusion_rate = config["print_speed"] * config["layer_height"] * extrusion_width

    # Add travel time estimate (roughly 30% of print time)
    print_time_seconds = total_volume / extrusion_rate * 1.3

    # Add time for heating, retraction, etc.
    overhead_seconds = layer_count * 2  # 2 seconds per layer overhead

    total_time_minutes = (print_time_seconds + overhead_seconds) / 60

    # Calculate filament weight (PLA density ~1.24 g/cm^3)
    filament_grams = (total_volume / 1000) * 1.24  # Convert mm^3 to cm^3

    return PrintEstimates(
        time_minutes=round(total_time_minutes, 1),
        filament_mm=round(filament_mm, 1),
        filament_grams=round(filament_grams, 2) if filament_grams > 0.01 else None,
        layer_height=config["layer_height"],
    )


def format_duration(minutes: float) -> str:
    """Format minutes into human-readable duration."""
    hours = int(minutes // 60)
    mins = int(minutes % 60)

    if hours > 0:
        return f"{hours}h {mins}min"
    return f"{mins}min"


def create_approval_message(
    estimates: PrintEstimates,
    model_name: str | None = None,
    include_keyboard: bool = True,
) -> ApprovalMessage:
    """
    Create a Telegram approval message with print estimates.

    Args:
        estimates: Print estimates from calculate_estimates()
        model_name: Optional model filename for display
        include_keyboard: Whether to include inline keyboard buttons

    Returns:
        ApprovalMessage with formatted text and optional keyboard
    """
    lines = []

    if model_name:
        lines.append(f"📦 *{model_name}*")
        lines.append("")

    lines.append("⏱ *Print Time:* " + format_duration(estimates.time_minutes))
    lines.append(f"🧵 *Filament:* {estimates.filament_mm:.1f}mm")

    if estimates.filament_grams:
        lines.append(f"⚖️ *Weight:* {estimates.filament_grams:.1f}g")

    if estimates.layer_height:
        lines.append(f"📐 *Layer Height:* {estimates.layer_height}mm")

    lines.append("")
    lines.append("Would you like to print this model?")

    text = "\n".join(lines)

    keyboard = None
    if include_keyboard:
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            keyboard = [
                [
                    InlineKeyboardButton("✅ Approve", callback_data="approve"),
                    InlineKeyboardButton("❌ Reject", callback_data="reject"),
                ],
                [
                    InlineKeyboardButton("⬆️ Scale Up", callback_data="scale_up"),
                    InlineKeyboardButton("⬇️ Scale Down", callback_data="scale_down"),
                ],
            ]
        except ImportError:
            logger.warning("python-telegram-bot not available, keyboard omitted")
            keyboard = None

    return ApprovalMessage(text=text, keyboard=keyboard)


def create_approval_keyboard(
    approve_callback: str = "approve",
    reject_callback: str = "reject",
    scale_up_callback: str = "scale_up",
    scale_down_callback: str = "scale_down",
) -> list[list["InlineKeyboardButton"]]:
    """
    Create inline keyboard for approval message.

    Args:
        approve_callback: Callback data for approve button
        reject_callback: Callback data for reject button
        scale_up_callback: Callback data for scale up button
        scale_down_callback: Callback data for scale down button

    Returns:
        2D list of InlineKeyboardButton for telegram
    """
    try:
        from telegram import InlineKeyboardButton

        return [
            [
                InlineKeyboardButton("✅ Approve", callback_data=approve_callback),
                InlineKeyboardButton("❌ Reject", callback_data=reject_callback),
            ],
            [
                InlineKeyboardButton("⬆️ Scale Up", callback_data=scale_up_callback),
                InlineKeyboardButton("⬇️ Scale Down", callback_data=scale_down_callback),
            ],
        ]
    except ImportError:
        logger.warning("python-telegram-bot not available")
        return []


def get_scale_callback_data(job_id: str, scale_factor: float) -> str:
    """
    Generate callback data for scale buttons with job ID.

    Args:
        job_id: Unique job identifier
        scale_factor: Scaling factor (e.g., 1.5 for 150%)

    Returns:
        Callback data string in format "scale:{job_id}:{scale_factor}"
    """
    return f"scale:{job_id}:{scale_factor}"


def parse_callback_data(callback_data: str) -> tuple[str, str, float | None]:
    """
    Parse callback data from inline keyboard buttons.

    Args:
        callback_data: The callback data string from Telegram

    Returns:
        Tuple of (action, job_id, scale_factor)
        scale_factor is None for approve/reject actions

    Raises:
        ValueError: If callback data format is invalid
    """
    parts = callback_data.split(":")

    if len(parts) < 2:
        raise ValueError(f"Invalid callback data format: {callback_data}")

    action = parts[0]
    job_id = parts[1]
    scale_factor = None

    if action == "scale" and len(parts) >= 3:
        try:
            scale_factor = float(parts[2])
        except ValueError:
            raise ValueError(f"Invalid scale factor in callback: {callback_data}")

    return action, job_id, scale_factor


async def send_approval_message(
    bot,
    chat_id: int,
    estimates: PrintEstimates,
    image_path: Path | str,
    job_id: str,
    model_name: str,
) -> int:
    """
    Send an approval message with inline keyboard to Telegram.

    Args:
        bot: Telegram Bot instance
        chat_id: Chat ID to send message to
        estimates: PrintEstimates with time and filament info
        image_path: Path to the thumbnail image
        job_id: Unique job identifier for callback data
        model_name: Name of the model for display

    Returns:
        Message ID of the sent message

    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If Telegram API call fails
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    keyboard = [
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{job_id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject:{job_id}"),
        ],
        [
            InlineKeyboardButton("🔍 Scale 150%", callback_data=f"scale:{job_id}:1.5"),
            InlineKeyboardButton("🔍 Scale 75%", callback_data=f"scale:{job_id}:0.75"),
        ],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    lines = []
    lines.append(f"📐 *{model_name}*\n")
    lines.append(f"⏱️ Print time: *{format_duration(estimates.time_minutes)}*")
    lines.append(f"🧵 Filament: *{estimates.filament_mm:.0f}mm*")
    if estimates.filament_grams:
        lines.append(f"⚖️ Weight: *{estimates.filament_grams:.1f}g*")
    lines.append(f"\nApprove to start printing?")

    caption = "\n".join(lines)

    with open(image_path, "rb") as photo:
        message = await bot.send_photo(
            chat_id=chat_id,
            photo=photo,
            caption=caption,
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )

    logger.info(f"Sent approval message {message.message_id} for job {job_id}")
    return message.message_id
