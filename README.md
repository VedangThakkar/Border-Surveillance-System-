# AI-Based Intelligent Border Surveillance System

This project is a lightweight demo of a border surveillance system built with Python, YOLOv8, OpenCV, and analytics tooling. It upgrades a basic object detection pipeline into a smarter intrusion-monitoring workflow with intruder tracking, border line crossing detection, risk scoring, alerting, and event analytics.

## Features

- YOLOv8-based real-time intruder detection
- ByteTrack-powered multi-frame tracking with unique track IDs
- Border line crossing detection for intrusion events
- Behavior analysis for walking, running, crawling, and loitering
- Frame-derived motion and low-light estimation instead of random sensor simulation
- Risk scoring based on behavior, border crossing, direction, group size, lighting, and motion
- Email alerts with track ID, risk level, timestamp, and optional snapshot
- CSV event logging for later reporting and dashboard integration
- Analytics generation for risk distribution, behavior trends, crossings by hour, and heatmap-style plots
- Multi-camera-ready configuration structure

## Project Structure

```text
Microsoft/
|-- main.py
|-- alert_system.py
|-- logger.py
|-- analytics.py
|-- requirements.txt
|-- README.md
|-- snapshots/
|-- analytics_output/
```

## Tech Stack

- Python 3.11
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Python Dotenv

## How It Works

1. `main.py` opens the configured camera feed.
2. YOLOv8 detects people and ByteTrack assigns persistent IDs.
3. Each tracked intruder is monitored across frames.
4. The system checks whether the intruder crosses the configured border line.
5. Movement history is used to estimate behavior and direction.
6. Frame brightness and motion are used as lightweight scene intelligence.
7. A dynamic risk score is calculated.
8. High-risk border events trigger email alerts and snapshots.
9. All events are logged for analytics and dashboard integration.

## Setup

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Create a `.env` file

Create a `.env` file in the project root with:

```env
OWNER_EMAIL=your_email@gmail.com
APP_PASSWORD=your_gmail_app_password
TO_EMAIL=receiver_email@gmail.com
LIVE_FEED_LINK=http://localhost/live-feed
MODEL_PATH=yolov8n.pt
```

## Run the Project

```powershell
python main.py
```

When running for the first time, YOLOv8 may download `yolov8n.pt` automatically if it is not already present.

## Configuration Notes

The camera and border line setup lives in `main.py` inside the `CAMERAS` list.

Example:

```python
CAMERAS = [
    CameraConfig(
        camera_id="cam_1",
        source=0,
        border_line=((120, 280), (560, 280)),
        name="Main Border Gate",
    ),
]
```

You can change:

- `source` to a webcam index or RTSP URL
- `border_line` to match your demo scene
- thresholds for confidence, cooldown, loitering, and grouping using environment variables or constants in `main.py`

## Generated Outputs

- `logs.csv`: structured surveillance event logs
- `snapshots/`: saved alert images
- `analytics_output/risk_distribution.png`
- `analytics_output/behavior_frequency.png`
- `analytics_output/border_crossings_by_hour.png`
- `analytics_output/intrusion_heatmap.png`

## Analytics

To generate analytics from logged events:

```powershell
python analytics.py
```

## Current Demo Scope

This version is designed as a demo and proof of concept:

- lightweight enough for a laptop
- simple single-process pipeline
- border line is manually configured
- tracking is optimized for real-time demo performance

## Suggested Next Improvements

- integrate MySQL logging directly instead of CSV-only logging
- connect events to the PHP dashboard in real time
- add animal filtering for false alert reduction
- add true parallel multi-camera processing
- support custom trained YOLO weights for defense-specific targets
- export intrusion summaries for command/control dashboards

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### ByteTrack or `lap` error on first run

Restart the Python process and run:

```powershell
python main.py
```

### Email alerts not sending

- confirm `.env` values are set correctly
- use a valid Gmail app password
- check internet connectivity and SMTP access

## License

This project is currently a demo/prototype for academic or portfolio use. Add your preferred license before publishing publicly.
