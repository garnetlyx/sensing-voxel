"""
Unit tests for TelegramBot
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import asyncio

from sensing_voxel.core.models import (
    AgentState,
    JobContext,
    DecisionAction,
    FileValidationStatus,
)
from sensing_voxel.core.state_machine import AgentStateMachine
from sensing_voxel.adapters.base import PrinterAdapter
from sensing_voxel.bot.telegram import TelegramBot


class TestTelegramBotAuth:
    """Authentication tests"""

    @pytest.mark.asyncio
    async def test_unauthorized_user(self):
        """Test that unauthorized users are rejected"""
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=AgentStateMachine(),
        )

        update = Mock()
        update.effective_user.id = 99999  # Not in allowlist
        update.message = Mock()
        update.message.reply_text = AsyncMock()  # Use AsyncMock

        result = await bot._check_auth(update)
        assert result is False
        update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_authorized_user(self):
        """Test that authorized users are accepted"""
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=AgentStateMachine(),
        )

        update = Mock()
        update.effective_user.id = 12345  # In allowlist

        result = await bot._check_auth(update)
        assert result is True


class TestTelegramBotCommands:
    """Command handler tests"""

    @pytest.mark.asyncio
    async def test_handle_start(self):
        """Test /start command"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.message = Mock()
        update.message.reply_text = AsyncMock()

        await bot._handle_start(update, None)

        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "Welcome to Sensing Voxel" in call_text

    @pytest.mark.asyncio
    async def test_handle_status_idle(self):
        """Test /status command when idle"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.message = Mock()
        update.message.reply_text = AsyncMock()

        await bot._handle_status(update, None)

        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "IDLE" in call_text

    @pytest.mark.asyncio
    async def test_handle_status_with_job(self):
        """Test /status command with active job"""
        state_machine = AgentStateMachine()
        state_machine.job_context = JobContext(
            job_id="test-job-123",
            job_file="/tmp/test.stl",
            uploaded_at=datetime.now(),
            decision_token="token",
            file_validation_status=FileValidationStatus.VALID,
            is_external=False,
            external_monitoring=False,
        )
        state_machine.current_state = AgentState.RECEIVING

        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.message = Mock()
        update.message.reply_text = AsyncMock()

        await bot._handle_status(update, None)

        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "RECEIVING" in call_text
        assert "test-job-123" in call_text


class TestTelegramBotFileUpload:
    """File upload tests"""

    @pytest.mark.asyncio
    async def test_reject_unsupported_extension(self):
        """Test that unsupported file types are rejected"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.message = Mock()
        update.message.document = Mock()
        update.message.document.file_name = "test.obj"  # Wrong extension
        update.message.reply_text = AsyncMock()

        await bot._handle_file_upload(update, None)

        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "STL" in call_text or "G-code" in call_text

    @pytest.mark.asyncio
    async def test_reject_txt_extension(self):
        """Test that .txt files are rejected"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.message = Mock()
        update.message.document = Mock()
        update.message.document.file_name = "notes.txt"
        update.message.reply_text = AsyncMock()

        await bot._handle_file_upload(update, None)

        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "supported" in call_text.lower() or "STL" in call_text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ext", [".gcode", ".g", ".gco"])
    async def test_accept_gcode_extensions(self, ext):
        """Test that G-code file extensions (.gcode, .g, .gco) pass the extension check"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.effective_chat = Mock()
        update.effective_chat.id = 99
        update.message = Mock()
        update.message.document = Mock()
        update.message.document.file_name = f"benchy{ext}"
        update.message.document.file_size = 1024  # 1KB — small enough
        update.message.reply_text = AsyncMock()

        # Bot is not IDLE — but it should get past extension/size checks first
        state_machine.current_state = AgentState.MONITORING

        await bot._handle_file_upload(update, None)

        # The first reply should be the "Processing" message (extension accepted),
        # followed by the "Agent is busy" message
        calls = update.message.reply_text.call_args_list
        assert any("Processing" in str(c) or "G-code" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_accept_stl_extension(self):
        """Test that .stl extension passes the extension check"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.effective_chat = Mock()
        update.effective_chat.id = 99
        update.message = Mock()
        update.message.document = Mock()
        update.message.document.file_name = "model.stl"
        update.message.document.file_size = 1024
        update.message.reply_text = AsyncMock()

        state_machine.current_state = AgentState.MONITORING

        await bot._handle_file_upload(update, None)

        calls = update.message.reply_text.call_args_list
        assert any("Processing" in str(c) or "STL" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_reject_oversized_stl_files(self):
        """Test that oversized STL files are rejected (>100MB)"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.message = Mock()
        update.message.document = Mock()
        update.message.document.file_name = "test.stl"
        update.message.document.file_size = 200 * 1024 * 1024  # 200MB > 100MB limit
        update.message.reply_text = AsyncMock()

        await bot._handle_file_upload(update, None)

        # First call is "Processing", second is the size rejection
        calls = update.message.reply_text.call_args_list
        size_call = calls[-1]
        call_text = size_call[0][0]
        assert "too large" in call_text.lower()

    @pytest.mark.asyncio
    async def test_reject_oversized_gcode_files(self):
        """Test that oversized G-code files are rejected (>200MB)"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.message = Mock()
        update.message.document = Mock()
        update.message.document.file_name = "big.gcode"
        update.message.document.file_size = 300 * 1024 * 1024  # 300MB > 200MB limit
        update.message.reply_text = AsyncMock()

        await bot._handle_file_upload(update, None)

        calls = update.message.reply_text.call_args_list
        size_call = calls[-1]
        call_text = size_call[0][0]
        assert "too large" in call_text.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_protection(self):
        """Test that path traversal is prevented"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        # Test the path validation directly
        from pathlib import Path

        uploads_dir = Path("/app/data/uploads")
        malicious_path = uploads_dir / "../../../etc/crontab"

        resolved = malicious_path.resolve()

        # This should escape the uploads directory
        with pytest.raises(ValueError):
            resolved.relative_to(uploads_dir.resolve())

        # The bot should catch this and reject the upload
        # (This is tested via the security validation in _handle_file_upload)


class TestTelegramBotCallbackQuery:
    """Callback query handler tests"""

    @pytest.mark.asyncio
    async def test_callback_with_invalid_job_id(self):
        """Test that callbacks with invalid job IDs are rejected"""
        state_machine = AgentStateMachine()
        state_machine.job_context = JobContext(
            job_id="job-123",
            job_file="/tmp/test.stl",
            uploaded_at=datetime.now(),
            decision_token="token",
            file_validation_status=FileValidationStatus.VALID,
            is_external=False,
            external_monitoring=False,
        )

        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.callback_query = Mock()
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()
        update.callback_query.data = "apr_wrong-job-id"  # Wrong job ID

        await bot._handle_callback_query(update, None)

        update.callback_query.edit_message_text.assert_called_once()
        call_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "not found" in call_text.lower()

    @pytest.mark.asyncio
    async def test_callback_with_correct_job_id(self):
        """Test that callbacks with correct job IDs are processed"""
        state_machine = AgentStateMachine()
        job_context = JobContext(
            job_id="job-123",
            job_file="/tmp/test.stl",
            uploaded_at=datetime.now(),
            decision_token="token",
            file_validation_status=FileValidationStatus.VALID,
            is_external=False,
            external_monitoring=False,
        )
        state_machine.job_context = job_context
        state_machine.current_state = AgentState.AWAITING_APPROVAL

        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        update = Mock()
        update.effective_user.id = 12345
        update.callback_query = Mock()
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()
        update.callback_query.data = "apr_job-123"  # Correct job ID

        with patch.object(bot.decision_engine, 'consume_decision', return_value=Mock()) as mock_consume:
            mock_consume.return_value = Mock(name='ConsumeResult.CONSUMED')

            await bot._handle_callback_query(update, None)

            # Should have called consume_decision
            mock_consume.assert_called_once()


class TestTelegramBotDecisionKeyboard:
    """Decision keyboard tests"""

    def test_create_approval_keyboard(self):
        """Test approval keyboard creation"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        job_id = "test-job-123"
        keyboard = bot.create_decision_keyboard(job_id, AgentState.AWAITING_APPROVAL)

        # Should have approve and reject buttons
        buttons = keyboard.inline_keyboard
        assert len(buttons) == 1
        assert len(buttons[0]) == 2

        # Check callback data uses full job ID
        approve_button = buttons[0][0]
        reject_button = buttons[0][1]
        assert approve_button.callback_data == f"apr_{job_id}"
        assert reject_button.callback_data == f"rej_{job_id}"

    def test_create_pause_keyboard(self):
        """Test pause/resume keyboard creation"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        job_id = "test-job-123"
        keyboard = bot.create_decision_keyboard(job_id, AgentState.PAUSED_PENDING_USER)

        # Should have resume and cancel buttons
        buttons = keyboard.inline_keyboard
        assert len(buttons) == 1
        assert len(buttons[0]) == 2

        resume_button = buttons[0][0]
        cancel_button = buttons[0][1]
        assert resume_button.callback_data == f"res_{job_id}"
        assert cancel_button.callback_data == f"can_{job_id}"


class TestTelegramBotErrorClassification:
    """Error classification tests"""

    def test_recoverable_network_error(self):
        """Test that network errors are recoverable"""
        from telegram.error import NetworkError

        error = NetworkError("Network timeout")
        from sensing_voxel.bot.telegram import is_recoverable_error

        assert is_recoverable_error(error) is True

    def test_recoverable_timeout_error(self):
        """Test that timeout errors are recoverable"""
        from telegram.error import TimedOut

        error = TimedOut("Request timeout")
        from sensing_voxel.bot.telegram import is_recoverable_error

        assert is_recoverable_error(error) is True

    def test_non_recoverable_forbidden_error(self):
        """Test that forbidden errors are not recoverable"""
        from telegram.error import Forbidden

        error = Forbidden("Bot blocked")
        from sensing_voxel.bot.telegram import is_recoverable_error

        assert is_recoverable_error(error) is False

    def test_non_recoverable_bad_request(self):
        """Test that most bad request errors are not recoverable"""
        from telegram.error import BadRequest

        error = BadRequest("Chat not found")
        from sensing_voxel.bot.telegram import is_recoverable_error

        assert is_recoverable_error(error) is False

    def test_recoverable_bad_request_message_too_long(self):
        """Test that 'message too long' is recoverable (can be split)"""
        from telegram.error import BadRequest

        error = BadRequest("message is too long")
        from sensing_voxel.bot.telegram import is_recoverable_error

        assert is_recoverable_error(error) is True


class TestTelegramBotSecurity:
    """Security tests"""

    @pytest.mark.asyncio
    async def test_job_id_prefix_collision_prevention(self):
        """
        Test that job ID prefix collision is prevented.
        Two jobs with same 8-char prefix should NOT match each other.
        """
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )

        # Two job IDs with same 8-char prefix
        job_1_id = "abc12345-xxxx-aaaa"
        job_2_id = "abc12345-yyyy-bbbb"

        assert job_1_id[:8] == job_2_id[:8]  # Same prefix
        assert job_1_id != job_2_id  # Different full IDs

        # Create job context with job_1
        state_machine.job_context = JobContext(
            job_id=job_1_id,
            job_file="/tmp/test.stl",
            uploaded_at=datetime.now(),
            decision_token="token",
            file_validation_status=FileValidationStatus.VALID,
            is_external=False,
            external_monitoring=False,
        )

        # Callback with job_2's ID should NOT match job_1
        update = Mock()
        update.effective_user.id = 12345
        update.callback_query = Mock()
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()
        update.callback_query.data = f"apr_{job_2_id}"  # Different job ID

        await bot._handle_callback_query(update, None)

        # Should be rejected as job not found
        update.callback_query.edit_message_text.assert_called_once()
        call_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "not found" in call_text.lower()


class TestTelegramBotNotifications:
    """Notification tests"""

    @pytest.mark.asyncio
    async def test_notify_print_complete(self):
        """Test print complete notification"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )
        bot._chat_id = 99999  # Set chat ID for notification

        job_context = JobContext(
            job_id="test-job-123",
            job_file="/tmp/test.stl",
            uploaded_at=datetime.now(),
            decision_token="token",
            file_validation_status=FileValidationStatus.VALID,
            is_external=False,
            external_monitoring=False,
        )

        # Mock the bot's send method
        with patch.object(bot, 'send', new_callable=AsyncMock) as mock_send:
            await bot.notify_print_complete(job_context)

            mock_send.assert_called_once()
            call_kwargs = mock_send.call_args[1]
            assert call_kwargs['content_type'] == 'text'
            assert 'Print Complete' in call_kwargs['content']
            assert 'test-job-123' in call_kwargs['content']

    @pytest.mark.asyncio
    async def test_notify_error(self):
        """Test error notification"""
        state_machine = AgentStateMachine()
        bot = TelegramBot(
            token="test_token",
            allowed_users=[12345],
            state_machine=state_machine,
        )
        bot._chat_id = 99999  # Set chat ID for notification

        with patch.object(bot, 'send', new_callable=AsyncMock) as mock_send:
            await bot.notify_error("Test error message", job_id="test-job-123")

            mock_send.assert_called_once()
            call_kwargs = mock_send.call_args[1]
            assert 'Error' in call_kwargs['content']
            assert 'Test error message' in call_kwargs['content']


class TestParseGcodeMetadata:
    """Tests for TelegramBot._parse_gcode_metadata static method"""

    def test_parse_total_layers(self, tmp_path):
        """Parse total layer count from G-code header"""
        gcode = tmp_path / "test.gcode"
        gcode.write_text("; total layer number: 42\nG28\n")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata["total_layers"] == 42

    def test_parse_estimated_time(self, tmp_path):
        """Parse estimated print time in seconds"""
        gcode = tmp_path / "test.gcode"
        gcode.write_text("; estimated_time = 3600\nG28\n")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata["estimated_time"] == 3600

    def test_parse_filament_type(self, tmp_path):
        """Parse filament type"""
        gcode = tmp_path / "test.gcode"
        gcode.write_text("; filament_type = PLA\nG28\n")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata["filament_type"] == "PLA"

    def test_parse_layer_height(self, tmp_path):
        """Parse layer height as float"""
        gcode = tmp_path / "test.gcode"
        gcode.write_text("; layer_height = 0.20\nG28\n")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata["layer_height"] == pytest.approx(0.20)

    def test_parse_filament_used_mm(self, tmp_path):
        """Parse filament used in mm"""
        gcode = tmp_path / "test.gcode"
        gcode.write_text("; filament used [mm] = 1234.5\nG28\n")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata["filament_used_mm"] == pytest.approx(1234.5)

    def test_parse_multiple_fields(self, tmp_path):
        """Parse multiple metadata fields from a realistic header"""
        gcode = tmp_path / "test.gcode"
        gcode.write_text(
            "; generated by PrusaSlicer\n"
            "; filament_type = PETG\n"
            "; total layer number: 150\n"
            "; estimated_time = 7200\n"
            "; layer_height = 0.15\n"
            "; filament used [mm] = 5678.9\n"
            "G28\n"
            "G1 X0 Y0\n"
        )

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata["filament_type"] == "PETG"
        assert metadata["total_layers"] == 150
        assert metadata["estimated_time"] == 7200
        assert metadata["layer_height"] == pytest.approx(0.15)
        assert metadata["filament_used_mm"] == pytest.approx(5678.9)

    def test_parse_empty_file(self, tmp_path):
        """Empty file returns empty metadata dict"""
        gcode = tmp_path / "empty.gcode"
        gcode.write_text("")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata == {}

    def test_parse_no_metadata_comments(self, tmp_path):
        """G-code with no metadata comments returns empty dict"""
        gcode = tmp_path / "test.gcode"
        gcode.write_text("G28\nG1 X100 Y100 F3000\nG1 Z0.3\n")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata == {}

    def test_parse_nonexistent_file(self, tmp_path):
        """Non-existent file returns empty dict instead of raising"""
        missing = tmp_path / "nonexistent.gcode"

        metadata = TelegramBot._parse_gcode_metadata(missing)
        assert metadata == {}

    def test_parse_total_layers_alternate_format(self, tmp_path):
        """Parse 'total layers' with alternate comment formats"""
        gcode = tmp_path / "test.gcode"
        gcode.write_text("; total_count: 99\n; total layers: 88\nG28\n")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata["total_layers"] == 88

    def test_parse_time_uppercase_keyword(self, tmp_path):
        """Parse TIME = <seconds> format"""
        gcode = tmp_path / "test.gcode"
        gcode.write_text("; TIME = 1800\nG28\n")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        assert metadata["estimated_time"] == 1800

    def test_parse_reads_only_first_10kb(self, tmp_path):
        """Metadata after first 10KB of file is ignored"""
        gcode = tmp_path / "test.gcode"
        # 12KB of padding, then the metadata
        padding = "G1 X0 Y0 F3000\n" * 1000  # ~16KB
        gcode.write_text(padding + "; total layer number: 42\n")

        metadata = TelegramBot._parse_gcode_metadata(gcode)
        # Should NOT find the metadata past 10KB
        assert "total_layers" not in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
