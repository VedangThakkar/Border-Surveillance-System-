import csv
import os
from datetime import datetime

LOG_FILE = "logs.csv"

LOG_HEADERS = [
    "timestamp",
    "camera_id",
    "track_id",
    "object",
    "confidence",
    "risk",
    "risk_score",
    "motion",
    "brightness",
    "temperature_proxy",
    "infrared_proxy",
    "behavior",
    "crossed_border",
    "group_size",
    "speed",
    "border_distance",
]


def initialize_logger() -> None:
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(LOG_HEADERS)


def log_event(
    camera_id: str,
    track_id: int,
    object_name: str,
    confidence: float,
    risk: str,
    risk_score: int,
    sensor_data: dict,
    behavior: str,
    crossed_border: bool,
    group_size: int,
    speed: float,
    border_distance: float,
) -> None:
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                camera_id,
                track_id,
                object_name,
                round(float(confidence), 3),
                risk,
                risk_score,
                sensor_data["motion"],
                sensor_data["brightness"],
                sensor_data["temperature_proxy"],
                sensor_data["infrared_proxy"],
                behavior,
                crossed_border,
                group_size,
                speed,
                border_distance,
            ]
        )
