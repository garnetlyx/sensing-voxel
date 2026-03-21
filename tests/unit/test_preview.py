"""
Unit tests for preview module.

Tests 3D rendering, multi-angle views, print estimates, and Telegram
approval message generation.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import trimesh

from sensing_voxel.modeling.preview import (
    ApprovalMessage,
    PrintEstimates,
    calculate_estimates,
    create_approval_keyboard,
    create_approval_message,
    format_duration,
    get_scale_callback_data,
    parse_callback_data,
    render_multi_angle,
    render_thumbnail,
    send_approval_message,
)


class TestRenderThumbnail:
    """Tests for render_thumbnail function."""

    @pytest.fixture
    def sample_mesh(self, tmp_path: Path) -> Path:
        """Create a sample cube mesh for testing."""
        # Create a simple cube mesh
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Bottom
                [4, 7, 6],
                [4, 6, 5],  # Top
                [0, 4, 5],
                [0, 5, 1],  # Front
                [2, 6, 7],
                [2, 7, 3],  # Back
                [0, 3, 7],
                [0, 7, 4],  # Left
                [1, 5, 6],
                [1, 6, 2],  # Right
            ]
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_path = tmp_path / "test_cube.stl"
        mesh.export(mesh_path)
        return mesh_path

    def test_render_thumbnail_success(self, tmp_path: Path, sample_mesh: Path) -> None:
        """Test successful thumbnail rendering."""
        output_path = tmp_path / "thumbnail.png"

        result = render_thumbnail(sample_mesh, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_thumbnail_creates_directory(self, tmp_path: Path, sample_mesh: Path) -> None:
        """Test that thumbnail creates output directory if needed."""
        output_path = tmp_path / "sub" / "dir" / "thumbnail.png"

        result = render_thumbnail(sample_mesh, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_render_thumbnail_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for non-existent mesh."""
        with pytest.raises(FileNotFoundError) as exc_info:
            render_thumbnail(tmp_path / "nonexistent.stl", tmp_path / "out.png")

        assert "not found" in str(exc_info.value)

    def test_render_thumbnail_invalid_mesh(self, tmp_path: Path) -> None:
        """Test ValueError for invalid mesh file."""
        # Create an empty file
        invalid_mesh = tmp_path / "invalid.stl"
        invalid_mesh.write_text("")

        with pytest.raises(ValueError) as exc_info:
            render_thumbnail(invalid_mesh, tmp_path / "out.png")

        assert "Failed to load" in str(exc_info.value)


class TestRenderMultiAngle:
    """Tests for render_multi_angle function."""

    @pytest.fixture
    def sample_mesh(self, tmp_path: Path) -> Path:
        """Create a sample cube mesh for testing."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 7, 6],
                [4, 6, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 6, 7],
                [2, 7, 3],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 6],
                [1, 6, 2],
            ]
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_path = tmp_path / "test_cube.stl"
        mesh.export(mesh_path)
        return mesh_path

    def test_render_multi_angle_default_views(self, tmp_path: Path, sample_mesh: Path) -> None:
        """Test rendering default multi-angle views."""
        output_dir = tmp_path / "previews"

        result = render_multi_angle(sample_mesh, output_dir)

        assert len(result) == 3
        for path in result:
            assert path.exists()
            assert path.suffix == ".png"

        # Check that all three views are created
        assert any("front" in p.name for p in result)
        assert any("side" in p.name for p in result)
        assert any("top" in p.name for p in result)

    def test_render_multi_angle_custom_views(self, tmp_path: Path, sample_mesh: Path) -> None:
        """Test rendering custom views."""
        output_dir = tmp_path / "previews"
        custom_views = [
            (45, 45, "custom1"),
            (90, 0, "custom2"),
        ]

        result = render_multi_angle(sample_mesh, output_dir, views=custom_views)

        assert len(result) == 2
        assert any("custom1" in p.name for p in result)
        assert any("custom2" in p.name for p in result)

    def test_render_multi_angle_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for non-existent mesh."""
        with pytest.raises(FileNotFoundError):
            render_multi_angle(tmp_path / "nonexistent.stl", tmp_path / "out")


class TestCalculateEstimates:
    """Tests for calculate_estimates function."""

    @pytest.fixture
    def sample_mesh(self, tmp_path: Path) -> Path:
        """Create a sample 10mm cube mesh for testing."""
        # 10mm cube
        vertices = np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [10, 10, 0],
                [0, 10, 0],
                [0, 0, 10],
                [10, 0, 10],
                [10, 10, 10],
                [0, 10, 10],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 7, 6],
                [4, 6, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 6, 7],
                [2, 7, 3],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 6],
                [1, 6, 2],
            ]
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_path = tmp_path / "test_cube.stl"
        mesh.export(mesh_path)
        return mesh_path

    def test_calculate_estimates_basic(self, tmp_path: Path, sample_mesh: Path) -> None:
        """Test basic estimate calculation."""
        result = calculate_estimates(sample_mesh)

        assert isinstance(result, PrintEstimates)
        assert result.time_minutes > 0
        assert result.filament_mm > 0
        assert result.layer_height == 0.2

    def test_calculate_estimates_with_config(self, tmp_path: Path, sample_mesh: Path) -> None:
        """Test estimate calculation with custom slicer config."""
        config = {
            "layer_height": 0.1,
            "print_speed": 100,
            "infill_density": 0.5,
        }

        result = calculate_estimates(sample_mesh, slicer_config=config)

        assert result.layer_height == 0.1
        # Higher infill should give more filament
        assert result.filament_mm > 0

    def test_calculate_estimates_returns_filament_grams(
        self, tmp_path: Path, sample_mesh: Path
    ) -> None:
        """Test that filament grams is calculated."""
        result = calculate_estimates(sample_mesh)

        assert result.filament_grams is not None
        assert result.filament_grams > 0

    def test_calculate_estimates_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for non-existent mesh."""
        with pytest.raises(FileNotFoundError):
            calculate_estimates(tmp_path / "nonexistent.stl")

    def test_calculate_estimates_empty_mesh(self, tmp_path: Path) -> None:
        """Test ValueError for empty mesh."""
        empty_mesh = tmp_path / "empty.stl"
        empty_mesh.write_text("")

        with pytest.raises(ValueError):
            calculate_estimates(empty_mesh)


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_duration_minutes_only(self) -> None:
        """Test formatting minutes only."""
        assert format_duration(45) == "45min"
        assert format_duration(0) == "0min"
        assert format_duration(59) == "59min"

    def test_format_duration_with_hours(self) -> None:
        """Test formatting with hours."""
        assert format_duration(60) == "1h 0min"
        assert format_duration(90) == "1h 30min"
        assert format_duration(125) == "2h 5min"

    def test_format_duration_large_values(self) -> None:
        """Test formatting large durations."""
        assert format_duration(1440) == "24h 0min"
        assert format_duration(150) == "2h 30min"


class TestCreateApprovalMessage:
    """Tests for create_approval_message function."""

    @pytest.fixture
    def sample_estimates(self) -> PrintEstimates:
        """Create sample print estimates."""
        return PrintEstimates(
            time_minutes=90,
            filament_mm=5000,
            filament_grams=15.5,
            layer_height=0.2,
        )

    def test_create_approval_message_basic(self, sample_estimates: PrintEstimates) -> None:
        """Test basic approval message creation."""
        result = create_approval_message(sample_estimates, include_keyboard=False)

        assert isinstance(result, ApprovalMessage)
        assert "90min" in result.text or "1h 30min" in result.text
        assert "5000.0mm" in result.text or "5000mm" in result.text
        assert result.keyboard is None

    def test_create_approval_message_with_model_name(
        self, sample_estimates: PrintEstimates
    ) -> None:
        """Test approval message with model name."""
        result = create_approval_message(
            sample_estimates, model_name="test_model.stl", include_keyboard=False
        )

        assert "test_model.stl" in result.text

    def test_create_approval_message_with_keyboard(self, sample_estimates: PrintEstimates) -> None:
        """Test approval message includes keyboard when telegram available."""
        import sys

        mock_telegram = MagicMock()
        mock_button = MagicMock(
            side_effect=lambda text, callback_data: MagicMock(
                text=text, callback_data=callback_data
            )
        )
        mock_telegram.InlineKeyboardButton = mock_button

        with patch.dict(sys.modules, {"telegram": mock_telegram}):
            import importlib
            import sensing_voxel.modeling.preview as preview_module

            importlib.reload(preview_module)

            result = preview_module.create_approval_message(sample_estimates, include_keyboard=True)

        # Keyboard should be present when telegram is available
        assert result.keyboard is not None

    def test_create_approval_message_without_filament_grams(self) -> None:
        """Test approval message without filament grams."""
        estimates = PrintEstimates(
            time_minutes=30,
            filament_mm=1000,
            filament_grams=None,
            layer_height=0.2,
        )

        result = create_approval_message(estimates, include_keyboard=False)

        assert "30min" in result.text or "0h 30min" in result.text
        # Should not include weight line
        assert "Weight" not in result.text


class TestCreateApprovalKeyboard:
    """Tests for create_approval_keyboard function."""

    def test_create_approval_keyboard_import_error(self) -> None:
        """Test keyboard creation when telegram not available."""
        with patch.dict("sys.modules", {"telegram": None}):
            result = create_approval_keyboard()
            assert result == []

    def test_create_approval_keyboard_structure(self) -> None:
        """Test keyboard has correct structure."""
        import sys

        mock_button = MagicMock(
            side_effect=lambda text, callback_data: MagicMock(
                text=text, callback_data=callback_data
            )
        )
        mock_telegram = MagicMock()
        mock_telegram.InlineKeyboardButton = mock_button

        with patch.dict(sys.modules, {"telegram": mock_telegram}):
            import importlib
            import sensing_voxel.modeling.preview as preview_module

            importlib.reload(preview_module)

            result = preview_module.create_approval_keyboard()

        # 2x2 grid: 2 rows, 2 columns each
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_create_approval_keyboard_custom_callbacks(self) -> None:
        """Test keyboard with custom callback data."""
        import sys

        mock_button = MagicMock(
            side_effect=lambda text, callback_data: MagicMock(
                text=text, callback_data=callback_data
            )
        )
        mock_telegram = MagicMock()
        mock_telegram.InlineKeyboardButton = mock_button

        with patch.dict(sys.modules, {"telegram": mock_telegram}):
            import importlib
            import sensing_voxel.modeling.preview as preview_module

            importlib.reload(preview_module)

            result = preview_module.create_approval_keyboard(
                approve_callback="custom_approve",
                reject_callback="custom_reject",
                scale_up_callback="custom_up",
                scale_down_callback="custom_down",
            )

        # Verify callback data for each button
        assert result[0][0].callback_data == "custom_approve"
        assert result[0][1].callback_data == "custom_reject"
        assert result[1][0].callback_data == "custom_up"
        assert result[1][1].callback_data == "custom_down"


class TestIntegration:
    """Integration tests combining multiple functions."""

    @pytest.fixture
    def sample_mesh(self, tmp_path: Path) -> Path:
        """Create a sample mesh for integration testing."""
        vertices = np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [10, 10, 0],
                [0, 10, 0],
                [0, 0, 10],
                [10, 0, 10],
                [10, 10, 10],
                [0, 10, 10],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 7, 6],
                [4, 6, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 6, 7],
                [2, 7, 3],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 6],
                [1, 6, 2],
            ]
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_path = tmp_path / "integration_cube.stl"
        mesh.export(mesh_path)
        return mesh_path

    def test_full_preview_workflow(self, tmp_path: Path, sample_mesh: Path) -> None:
        """Test complete preview workflow: render + estimates + message."""
        output_dir = tmp_path / "previews"

        # Step 1: Render thumbnail
        thumb_path = output_dir / "thumbnail.png"
        render_thumbnail(sample_mesh, thumb_path)
        assert thumb_path.exists()

        # Step 2: Render multi-angle
        views = render_multi_angle(sample_mesh, output_dir)
        assert len(views) == 3
        for v in views:
            assert v.exists()

        # Step 3: Calculate estimates
        estimates = calculate_estimates(sample_mesh)
        assert estimates.time_minutes > 0
        assert estimates.filament_mm > 0

        # Step 4: Create approval message
        message = create_approval_message(
            estimates, model_name="integration_cube.stl", include_keyboard=False
        )
        assert "integration_cube.stl" in message.text
        assert (
            str(int(estimates.filament_mm)) in message.text
            or str(estimates.filament_mm) in message.text
        )

    def test_end_to_end_with_keyboard(self, tmp_path: Path, sample_mesh: Path) -> None:
        """Test end-to-end with keyboard mock."""
        import sys

        estimates = calculate_estimates(sample_mesh)

        mock_telegram = MagicMock()
        mock_button = MagicMock(
            side_effect=lambda text, callback_data: MagicMock(
                text=text, callback_data=callback_data
            )
        )
        mock_telegram.InlineKeyboardButton = mock_button

        with patch.dict(sys.modules, {"telegram": mock_telegram}):
            import importlib
            import sensing_voxel.modeling.preview as preview_module

            importlib.reload(preview_module)

            message = preview_module.create_approval_message(estimates, include_keyboard=True)

        assert isinstance(message, preview_module.ApprovalMessage)
        assert message.text is not None
        assert len(message.text) > 0


class TestGetScaleCallbackData:
    """Tests for get_scale_callback_data function."""

    def test_get_scale_callback_data(self) -> None:
        """Test generating scale callback data."""
        result = get_scale_callback_data("job123", 1.5)
        assert result == "scale:job123:1.5"

    def test_get_scale_callback_data_decimal(self) -> None:
        """Test with decimal scale factor."""
        result = get_scale_callback_data("job456", 0.75)
        assert result == "scale:job456:0.75"


class TestParseCallbackData:
    """Tests for parse_callback_data function."""

    def test_parse_approve_callback(self) -> None:
        """Test parsing approve callback."""
        action, job_id, scale = parse_callback_data("approve:job123")
        assert action == "approve"
        assert job_id == "job123"
        assert scale is None

    def test_parse_reject_callback(self) -> None:
        """Test parsing reject callback."""
        action, job_id, scale = parse_callback_data("reject:job456")
        assert action == "reject"
        assert job_id == "job456"
        assert scale is None

    def test_parse_scale_callback(self) -> None:
        """Test parsing scale callback."""
        action, job_id, scale = parse_callback_data("scale:job789:1.5")
        assert action == "scale"
        assert job_id == "job789"
        assert scale == 1.5

    def test_parse_invalid_callback(self) -> None:
        """Test parsing invalid callback raises error."""
        with pytest.raises(ValueError):
            parse_callback_data("invalid")

    def test_parse_invalid_scale(self) -> None:
        """Test parsing callback with invalid scale."""
        with pytest.raises(ValueError):
            parse_callback_data("scale:job123:not_a_number")


class TestSendApprovalMessage:
    """Tests for send_approval_message function."""

    @pytest.mark.asyncio
    async def test_send_approval_message_success(
        self,
        tmp_path: Path,
    ) -> None:
        """Test successful message sending."""
        vertices = np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [10, 10, 0],
                [0, 10, 0],
                [0, 0, 10],
                [10, 0, 10],
                [10, 10, 10],
                [0, 10, 10],
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 7, 6],
                [4, 6, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 6, 7],
                [2, 7, 3],
                [0, 3, 7],
                [0, 7, 4],
                [1, 5, 6],
                [1, 6, 2],
            ]
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_path = tmp_path / "test.stl"
        mesh.export(mesh_path)
        thumb_path = render_thumbnail(mesh_path, tmp_path / "thumb.png")

        estimates = calculate_estimates(mesh_path)

        mock_message = MagicMock()
        mock_message.message_id = 12345
        mock_bot = AsyncMock()
        mock_bot.send_photo = AsyncMock(return_value=mock_message)

        result = await send_approval_message(
            bot=mock_bot,
            chat_id=123456,
            estimates=estimates,
            image_path=thumb_path,
            job_id="job_test_123",
            model_name="test.stl",
        )

        assert result == 12345
        mock_bot.send_photo.assert_called_once()
        call_kwargs = mock_bot.send_photo.call_args.kwargs
        assert call_kwargs["chat_id"] == 123456
        assert "test.stl" in call_kwargs["caption"]
        assert call_kwargs["parse_mode"] == "Markdown"
        assert call_kwargs["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_send_approval_message_image_not_found(
        self,
        tmp_path: Path,
    ) -> None:
        """Test error on missing image file."""
        estimates = PrintEstimates(
            time_minutes=45.5,
            filament_mm=1234.5,
            filament_grams=3.68,
            layer_height=0.2,
        )
        mock_bot = AsyncMock()

        with pytest.raises(FileNotFoundError):
            await send_approval_message(
                bot=mock_bot,
                chat_id=123456,
                estimates=estimates,
                image_path=tmp_path / "nonexistent.png",
                job_id="job_test",
                model_name="test.stl",
            )
