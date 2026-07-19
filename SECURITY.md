# Security Policy

## Supported Versions

Currently supported versions for security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in Sensing Voxel, please report it privately to help us fix it before public disclosure.

### How to Report

**Preferred**: Use [GitHub's private security advisory feature](https://github.com/garnetlyx/sensing-voxel/security/advisories/new) — this keeps reports private and tracked.

Alternatively, send an email to **security@garnetlyx.com** with:

1. **Description**: Clear description of the vulnerability
2. **Impact**: What an attacker could do if they exploit it
3. **Steps to Reproduce**: Detailed steps to reproduce the issue
4. **Affected Versions**: Which versions are affected
5. **Suggested Fix**: If you have ideas for fixing it (optional)

### What to Expect

- **Acknowledgment**: We'll acknowledge receipt within 48 hours
- **Assessment**: We'll assess the severity and impact within 5 business days
- **Updates**: We'll keep you informed of progress toward a fix
- **Fix Timeline**:
  - Critical vulnerabilities: 7 days
  - High severity: 30 days
  - Medium/Low severity: 90 days
- **Credit**: We'll credit you in the security advisory (unless you prefer to remain anonymous)

### Security Best Practices for Users

When deploying Sensing Voxel:

1. **Environment Variables and Local Configuration**
   - Never commit `.env` or `config/local.yaml` to version control
   - Keep real hostnames, IP addresses, MAC addresses, user IDs, camera URLs, and hardware identifiers out of public examples and test fixtures
   - Use RFC 5737 addresses (`192.0.2.0/24`, `198.51.100.0/24`, or `203.0.113.0/24`) and synthetic identifiers in documentation and tests
   - Generate `SEARXNG_SECRET` and other deployment secrets locally; do not replace public placeholders with usable credentials
   - Use strong, unique tokens for Telegram bots and API keys
   - Rotate credentials immediately if they are exposed

2. **Hardware and Integration Tests**
   - Treat `tests/real_world/` and hardware E2E tests as opt-in tests that can contact configured devices and external services
   - Do not run hardware tests against public example addresses
   - Review evidence, logs, screenshots, and captured responses before sharing them because they may contain deployment identifiers or sensitive payloads

3. **Telegram Bot Security**
   - Use the `allowed_users` configuration to restrict bot access
   - Only add trusted user IDs to the allowlist
   - Revoke and regenerate bot token if it's exposed

4. **Network Security**
   - Run the dashboard behind a reverse proxy with HTTPS
   - Use firewall rules to restrict printer access
   - Don't expose printer ports directly to the internet

5. **Camera Security**
   - Use strong passwords for camera authentication
   - Place cameras on isolated network segments
   - Disable camera remote access if not needed

6. **LLM Integration**
   - Use API keys with minimal required permissions
   - Enable approval workflows for high-risk commands
   - Review tool audit logs regularly (`data/tool_audit.jsonl`)

7. **File Upload Safety**
   - The system validates uploaded G-code files
   - Path traversal protection is enabled by default
   - Emergency commands (M112, M410) are blocked

### Known Security Considerations

1. **Telegram Bot Token**: Store securely in `config/local.yaml` (gitignored)
2. **Printer Access**: Klipper/Moonraker API access grants full printer control
3. **LLM Shell Access**: If enabled, LLM can execute shell commands (disabled by default)
4. **Camera Feeds**: RTSP streams may contain visual privacy information

### Security Features

- **Telegram Allowlist**: Only authorized user IDs can interact with the bot
- **G-code Validation**: Dangerous commands are blocked (M112, M410, M80, M81, M23)
- **Path Traversal Protection**: Uploaded files are validated and sanitized
- **Decision Approval**: High-risk actions require explicit user approval
- **Audit Logging**: All LLM tool calls are logged to `data/tool_audit.jsonl`

## Disclosure Policy

When we receive a security report:

1. We'll work with you to understand and verify the issue
2. We'll develop and test a fix
3. We'll prepare a security advisory
4. We'll release the fix and publish the advisory
5. We'll credit the reporter (unless they prefer anonymity)

We follow responsible disclosure practices and typically allow 90 days before public disclosure.

## Security Updates

Security updates are released as:
- Patch versions (0.1.x) for minor fixes
- Minor versions (0.x.0) for significant security improvements

Subscribe to releases on GitHub to be notified of security updates.

## Contact

For security-related questions that don't involve reporting vulnerabilities, you can open a GitHub Discussion in the Security category.
