# GestureFlow — AI Virtual Mouse

> Control your Windows desktop with hand gestures using your webcam — no physical mouse required.

## Overview

GestureFlow uses **MediaPipe's Hand Landmarker Tasks API** and **OpenCV** to detect hand landmarks in real time via a standard webcam. It maps gesture states to Windows mouse actions through **PyAutoGUI**, rendered inside a modern **PySide6** desktop application.

## Architecture

```
GestureFlow
├── src/gestureflow/
│   ├── camera/        # Webcam capture (off-UI thread)
│   ├── tracking/      # MediaPipe Tasks Hand Landmarker wrapper
│   ├── gestures/      # Gesture classifiers (future)
│   ├── controller/    # Mouse / keyboard action dispatcher (future)
│   ├── calibration/   # User calibration pipeline (future)
│   ├── ui/            # PySide6 windows, widgets, workers
│   ├── config/        # Settings loader / validator
│   └── utils/         # Logging, helpers
├── assets/
│   ├── models/        # MediaPipe .task model files
│   └── icons/         # App icons
├── tests/             # pytest test suite
├── scripts/           # Windows install / launch helpers
├── main.py
├── requirements.txt
└── default_settings.json
```

## Requirements

| Requirement | Version |
|-------------|---------|
| Python      | 3.11+   |
| OpenCV      | 4.10.x  |
| MediaPipe   | 0.10.x  |
| PySide6     | 6.7.x   |
| PyAutoGUI   | 0.9.x   |
| NumPy       | 1.26.x  |

## Quick Start

```bat
# 1. Install dependencies (creates .venv automatically)
scripts\install.bat

# 2. Download the MediaPipe hand model
scripts\download_model.bat

# 3. Launch the app
scripts\launch.bat
```

## Manual Setup

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\download_model.py
python main.py
```

## Safety

- **PyAutoGUI fail-safe is always ON** — move your mouse to any corner to abort.
- No administrator privileges required.
- Camera processing runs on a background thread; the UI thread is never blocked.

## Development

```bat
# Run all tests
.venv\Scripts\activate
pytest tests/ -v

# Build a standalone .exe (optional)
scripts\build.bat
```

## License

MIT License — see [LICENSE](LICENSE) for details.
