# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- License changed from MIT to GPL-3.0-or-later to ensure open source ecosystem compatibility

### Added
- Initial public release preparation
- Community documentation (CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md)

## [0.1.0] - 2026-07-10

### Added
- Telegram bot interface with file upload and notification support
- Klipper/Moonraker adapter with REST and WebSocket connectivity
- Web dashboard with real-time status and WebSocket updates
- G-code validation and security hardening
- Basic LLM chat integration with Anthropic-compatible API
- Tapo PTZ camera support with ONVIF control
- State machine with JSON persistence
- Decision engine with approval workflow
- Docker deployment support with environment variable configuration

### Security
- Telegram user allowlist for bot access control
- G-code command validation (blocks dangerous commands)
- Path traversal protection for file uploads
- LLM tool call audit logging

[unreleased]: https://github.com/garnetlyx/sensing-voxel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/garnetlyx/sensing-voxel/releases/tag/v0.1.0
