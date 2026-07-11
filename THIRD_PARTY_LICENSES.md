# Third-Party Licenses

This project is licensed under GPL-3.0-or-later. The following third-party dependencies have compatible licenses:

## GPL-Compatible Dependencies

### LGPL-3.0 (Weak Copyleft - Compatible with GPL-3.0)

#### python-telegram-bot
- **License**: GNU Lesser General Public License v3.0 (LGPLv3)
- **Source**: https://github.com/python-telegram-bot/python-telegram-bot
- **Usage**: Telegram bot interface for remote monitoring and control
- **Compatibility**: LGPL allows use in GPL projects

#### opencv-python (bundled components)
- **OpenCV core**: Apache-2.0 (wrapper scripts: MIT)
- **FFmpeg** (bundled in wheels): LGPLv2.1
- **Qt5** (bundled in non-headless Linux wheels): LGPLv3
- **Source**: https://github.com/opencv/opencv-python
- **Usage**: Computer vision processing for first-layer detection and camera calibration
- **Note**: All bundled LGPL components are compatible with GPL-3.0

### Permissive Licenses (Fully Compatible)

The following dependencies use permissive licenses that are compatible with GPL-3.0:

| Package | License | Purpose |
|---------|---------|---------|
| **pyyaml** | MIT | Configuration file parsing |
| **aiohttp** | Apache-2.0 | Async HTTP client/server |
| **numpy** | BSD-3-Clause | Numerical computations |
| **scipy** | BSD-3-Clause | Scientific computing |
| **pydantic** | MIT | Data validation |
| **filelock** | MIT | File locking primitives |
| **paho-mqtt** | EPL-2.0 OR BSD-3-Clause | MQTT client (we use BSD path) |
| **anthropic** | MIT | LLM API client |
| **trimesh** | MIT | 3D mesh processing |

### Optional Dependencies (camera support)

| Package | License | Purpose |
|---------|---------|---------|
| **pytapo** | MIT | Tapo camera control |
| **onvif-zeep** | MIT | ONVIF PTZ camera control |

## License Compatibility Summary

All runtime dependencies are compatible with GPL-3.0:

- **8 dependencies** use fully permissive licenses (MIT/BSD/Apache-2.0)
- **2 dependencies** use LGPL (weak copyleft, explicitly compatible with GPL)
- **0 dependencies** use incompatible licenses

### What This Means for Integrations

**✅ Compatible projects (can use Sensing Voxel):**
- OrcaSlicer (GPL-3.0)
- PrusaSlicer (AGPL-3.0)
- Klipper (GPL-3.0)
- Moonraker (GPL-3.0)
- Mainsail (GPL-3.0)
- Any other GPL-3.0 or AGPL-3.0 licensed software

**❌ Incompatible projects (cannot integrate without open-sourcing):**
- Bambu Studio (proprietary/closed-source)
- Creality Print (proprietary/closed-source)
- Any proprietary/closed-source 3D printing software

### For Developers

If you want to integrate Sensing Voxel into your project:

1. **Open source projects with GPL-3.0 compatible licenses**: You can freely use and integrate this code
2. **Proprietary/closed-source projects**: You must either:
   - Open source your entire project under GPL-3.0 or compatible license, OR
   - Contact the maintainers for alternative licensing arrangements

## References

- GNU GPL-3.0 License: https://www.gnu.org/licenses/gpl-3.0.html
- GPL Compatibility Matrix: https://www.gnu.org/licenses/license-list.html
- LGPL and GPL Compatibility: https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility

## Verifying Dependency Licenses

To verify the licenses of all installed dependencies:

```bash
pip install pip-licenses
pip-licenses --format=markdown --order=license
```

Last updated: 2026-07-10
