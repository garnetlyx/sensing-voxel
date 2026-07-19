# Contributing to Sensing Voxel

Thank you for your interest in contributing to Sensing Voxel! This document provides guidelines for contributing to the project.

## Getting Started

### Development Environment

1. **Prerequisites**
   - Python 3.12 or higher
   - Git
   - A 3D printer running Klipper/Moonraker or Bambu firmware (for testing hardware integrations)

2. **Setup**

```bash
# Clone the repository
git clone https://github.com/garnetlyx/sensing-voxel.git
cd sensing-voxel

# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Optional: Install camera support
pip install -e ".[camera]"
```

3. **Configuration**

```bash
# Copy the example config and customize for your setup
cp config/default.yaml config/local.yaml
# Edit config/local.yaml with your Telegram bot token, printer details, etc.
```

## Development Workflow

### Running Tests

```bash
# Run all tests (excludes hardware integration tests by default)
pytest

# Run with coverage report
pytest --cov=sensing_voxel --cov-report=html

# Run specific test file
pytest tests/unit/test_state_machine.py

# Run hardware integration tests (requires live printer)
pytest -m live
```

### Code Style

We use automated formatters and linters to maintain consistent code style:

```bash
# Format code with Black
black src/ tests/

# Sort imports
ruff check --select I --fix src/ tests/

# Run linter
ruff check src/ tests/

# Type checking
mypy src/
```

**Style Requirements:**
- Line length: 100 characters
- Python version: 3.12+
- Follow PEP 8 conventions
- Use type hints for function signatures
- Write docstrings for public APIs

### Making Changes

1. **Create a branch**

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

2. **Make your changes**
   - Write clear, focused commits
   - Include tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Run pre-commit checks**

```bash
# Format and lint
black src/ tests/
ruff check --fix src/ tests/

# Run tests
pytest

# Check types
mypy src/
```

## Pull Request Process

1. **Before submitting**
   - Ensure all tests pass
   - Add tests for new features or bug fixes
   - Update relevant documentation (README, docstrings, etc.)
   - Run code formatters and linters
   - Verify your changes don't break existing functionality

2. **PR Description**
   - Clearly describe the problem or feature
   - Explain your solution approach
   - List any breaking changes
   - Reference related issues (e.g., "Fixes #123")

3. **Review Process**
   - PRs require at least one approval
   - Address reviewer feedback promptly
   - Keep PRs focused and reasonably sized
   - Rebase on main if needed to resolve conflicts

## Project Structure

```text
sensing-voxel/
├── src/sensing_voxel/
│   ├── main.py         # Runtime composition and lifecycle
│   ├── config.py       # Configuration models and loading
│   ├── core/           # State machine, decisions, persistence
│   ├── safety/         # Validation, preview, authorization, execution
│   ├── monitoring/     # Camera/LLM print monitoring
│   ├── adapters/       # Printer adapter contract and implementations
│   ├── bot/            # Telegram interface and authorization
│   ├── api/            # HTTP/WebSocket dashboard API
│   ├── dashboard/      # Browser UI assets
│   ├── llm/            # Chat, tools, approvals, context, and audit
│   ├── camera/         # ONVIF PTZ support
│   ├── slicer/         # Model processing support; not fully runtime-wired
│   ├── storage/        # Runtime persistence
│   └── infra/          # Shared infrastructure utilities
├── tests/              # Unit, integration, end-to-end, and live tests
├── config/             # Configuration files
└── docs/               # Documentation
```

## Testing Guidelines

### Unit Tests

- Place in `tests/unit/`
- Test individual components in isolation
- Use mocks for external dependencies
- Aim for high coverage of core logic

### Integration Tests

- Place in `tests/integration/`
- Test interactions between components
- May use test fixtures instead of live hardware
- Mark with `@pytest.mark.integration` if needed

### Hardware Tests

- Place in `tests/real_world/` or `tests/e2e/`
- Mark with `@pytest.mark.live`
- Document hardware requirements in test docstrings
- Ensure tests are safe (won't damage printers)

## Documentation

- Update `README.md` for user-facing changes
- Write docstrings for all public functions and classes
- Use Google-style docstring format
- Update `docs/ARCHITECTURE.md` for architectural changes
- Add inline comments for complex logic

## Reporting Issues

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal steps to reproduce the problem
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - OS (Linux, macOS, Windows)
   - Python version
   - Printer firmware (Klipper/Bambu)
   - Relevant configuration settings
6. **Logs**: Relevant error messages or log output

## Security Issues

**Do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, please email the maintainers directly. Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

See `SECURITY.md` for our security policy and response process.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Contributor License Agreement (CLA)

By contributing to Sensing Voxel, you agree that:

1. **Open Source License**: Your contributions are licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later) for public open source use.

2. **Additional Rights**: You grant the project maintainers perpetual, worldwide, non-exclusive, royalty-free rights to:
   - Use, modify, and distribute your contributions under alternative licenses
   - Integrate your contributions into commercial or closed-source versions
   - Change the project license for future versions

3. **Your Rights**: You retain copyright to your contributions and can use them however you wish in your own projects.

**Why we need this**: This allows the project to:
- Offer commercial licenses to companies that cannot use GPL (while keeping the community version GPL)
- Migrate to different licenses in the future if needed (e.g., AGPL for cloud service protection)
- Maintain flexibility while ensuring the open source community always has a GPL version

This is similar to CLAs used by Qt, MongoDB, and other dual-licensed projects.

## Questions?

- Check existing issues and discussions
- Read the documentation in `docs/`
- Join our community discussions (if applicable)

Thank you for contributing to Sensing Voxel!
