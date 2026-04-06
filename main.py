import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), ".ultralytics"))
from ultralytics import YOLO

from alert_system import send_alert_email
from logger import initialize_logger, log_event


@dataclass
class CameraConfig:
    camera_id: str
    source: int | str
    border_line: Tuple[Tuple[int, int], Tuple[int, int]]
    name: str = "Border Camera"


@dataclass
class TrackState:
    track_id: int
    camera_id: str
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    frames_seen: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_alert_time: float = 0.0
    has_crossed_border: bool = False
    last_side: Optional[int] = None
    max_speed: float = 0.0
    loitering_frames: int = 0
    movement_pixels: float = 0.0


CAMERAS = [
    CameraConfig(
        camera_id="cam_1",
        source=0,
        border_line=((120, 280), (560, 280)),
        name="Main Border Gate",
    ),
]

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
TRACKER_CONFIG = os.getenv("TRACKER_CONFIG", "bytetrack.yaml")
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))
MAX_TRACK_AGE_SECONDS = int(os.getenv("MAX_TRACK_AGE_SECONDS", "8"))
LOITERING_FRAME_THRESHOLD = int(os.getenv("LOITERING_FRAME_THRESHOLD", "90"))
GROUP_DISTANCE_THRESHOLD = int(os.getenv("GROUP_DISTANCE_THRESHOLD", "120"))

model = YOLO(MODEL_PATH)
track_registry: Dict[str, Dict[int, TrackState]] = {camera.camera_id: {} for camera in CAMERAS}


def point_side(point: Tuple[float, float], line: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
    (x1, y1), (x2, y2) = line
    px, py = point
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return 1 if cross > 0 else -1 if cross < 0 else 0


def line_distance(point: Tuple[float, float], line: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    return numerator / denominator if denominator else 0.0


def estimate_scene_metrics(frame: np.ndarray, previous_gray: Optional[np.ndarray]) -> Tuple[Dict[str, float], np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    if previous_gray is None:
        motion_ratio = 0.0
    else:
        diff = cv2.absdiff(previous_gray, gray)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_ratio = float(np.count_nonzero(motion_mask)) / motion_mask.size

    sensor_data = {
        "motion": round(motion_ratio, 3),
        "brightness": round(brightness, 2),
        "infrared_proxy": round(1.0 - min(brightness / 255.0, 1.0), 3),
        "temperature_proxy": round(20.0 + (brightness / 255.0) * 15.0, 2),
    }
    return sensor_data, gray


def classify_behavior(state: TrackState, border_line: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[str, float, float]:
    if len(state.history) < 2:
        return "observed", 0.0, 0.0

    start_x, start_y = state.history[0]
    end_x, end_y = state.history[-1]
    dx = end_x - start_x
    dy = end_y - start_y
    speed = math.hypot(dx, dy) / max(len(state.history) - 1, 1)
    direction_score = 0.0

    if dy != 0:
        border_y = (border_line[0][1] + border_line[1][1]) / 2
        if end_y > start_y and end_y >= border_y:
            direction_score = 1.0
        elif end_y > start_y:
            direction_score = 0.6

    if speed > 12:
        behavior = "running"
    elif speed < 2 and state.loitering_frames >= LOITERING_FRAME_THRESHOLD:
        behavior = "loitering"
    elif speed < 4 and abs(dx) > abs(dy):
        behavior = "crawling"
    else:
        behavior = "walking"

    return behavior, speed, direction_score


def count_group_size(current_track_id: int, states: Dict[int, TrackState]) -> int:
    if current_track_id not in states or not states[current_track_id].history:
        return 1

    center = states[current_track_id].history[-1]
    group_count = 1
    for other_id, other_state in states.items():
        if other_id == current_track_id or not other_state.history:
            continue
        other_center = other_state.history[-1]
        distance = math.hypot(center[0] - other_center[0], center[1] - other_center[1])
        if distance <= GROUP_DISTANCE_THRESHOLD:
            group_count += 1
    return group_count


def calculate_risk(
    class_name: str,
    behavior: str,
    speed: float,
    direction_score: float,
    group_size: int,
    crossed_border: bool,
    sensor_data: Dict[str, float],
    confidence: float,
) -> Tuple[str, int]:
    score = 0

    if class_name == "Intruder":
        score += 3
    elif class_name in {"fire", "smoke", "brassknuckles", "switchblades"}:
        score += 4
    else:
        score += 1

    if crossed_border:
        score += 5
    if behavior == "running":
        score += 3
    elif behavior == "loitering":
        score += 2
    elif behavior == "crawling":
        score += 3

    if speed > 10:
        score += 2
    if direction_score >= 1.0:
        score += 2
    elif direction_score >= 0.5:
        score += 1

    if group_size >= 3:
        score += 3
    elif group_size == 2:
        score += 1

    if sensor_data["motion"] > 0.08:
        score += 1
    if sensor_data["infrared_proxy"] > 0.6:
        score += 2
    if sensor_data["brightness"] < 70:
        score += 2
    if confidence < 0.45:
        score -= 1

    hour = datetime.now().hour
    if hour >= 19 or hour <= 5:
        score += 2

    if score >= 11:
        return "HIGH", score
    if score >= 6:
        return "MEDIUM", score
    return "LOW", score


def should_trigger_alert(state: TrackState, risk_level: str, crossed_border: bool) -> bool:
    now = time.time()
    if risk_level != "HIGH":
        return False
    if not crossed_border and state.loitering_frames < LOITERING_FRAME_THRESHOLD:
        return False
    if now - state.last_alert_time < ALERT_COOLDOWN_SECONDS:
        return False
    return True


def annotate_frame(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    track_id: int,
    risk_level: str,
    risk_score: int,
    behavior: str,
    crossed_border: bool,
) -> None:
    x1, y1, x2, y2 = box
    color = (0, 0, 255) if risk_level == "HIGH" else (0, 255, 255) if risk_level == "MEDIUM" else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id} | {behavior} | {risk_level} ({risk_score})"
    if crossed_border:
        label += " | BORDER CROSS"
    cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def cleanup_stale_tracks(states: Dict[int, TrackState]) -> None:
    now = time.time()
    stale_ids = [track_id for track_id, state in states.items() if now - state.last_seen > MAX_TRACK_AGE_SECONDS]
    for track_id in stale_ids:
        del states[track_id]


def process_camera(camera: CameraConfig) -> None:
    cap = cv2.VideoCapture(camera.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Could not open source for {camera.name}: {camera.source}")
        return

    previous_gray = None
    window_name = f"{camera.name} [{camera.camera_id}]"

    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"Frame read failed for {camera.camera_id}")
            break

        sensor_data, previous_gray = estimate_scene_metrics(frame, previous_gray)
        states = track_registry[camera.camera_id]
        cleanup_stale_tracks(states)

        results = model.track(
            source=frame,
            persist=True,
            verbose=False,
            tracker=TRACKER_CONFIG,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],
        )

        cv2.line(frame, camera.border_line[0], camera.border_line[1], (255, 0, 0), 2)
        cv2.putText(frame, f"{camera.name}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for box, confidence, track_id in zip(boxes, confidences, track_ids):
                x1, y1, x2, y2 = box.tolist()
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                state = states.get(track_id)
                if state is None:
                    state = TrackState(track_id=track_id, camera_id=camera.camera_id)
                    states[track_id] = state

                state.frames_seen += 1
                state.last_seen = time.time()
                state.history.append(center)

                if len(state.history) > 1:
                    previous_point = state.history[-2]
                    movement = math.hypot(center[0] - previous_point[0], center[1] - previous_point[1])
                    state.movement_pixels += movement
                    state.max_speed = max(state.max_speed, movement)
                    if movement < 2.0:
                        state.loitering_frames += 1
                    else:
                        state.loitering_frames = max(0, state.loitering_frames - 1)

                current_side = point_side(center, camera.border_line)
                crossed_border = False
                if state.last_side is not None and current_side != 0 and current_side != state.last_side:
                    crossed_border = True
                    state.has_crossed_border = True
                state.last_side = current_side

                behavior, speed, direction_score = classify_behavior(state, camera.border_line)
                group_size = count_group_size(track_id, states)
                risk_level, risk_score = calculate_risk(
                    class_name="Intruder",
                    behavior=behavior,
                    speed=speed,
                    direction_score=direction_score,
                    group_size=group_size,
                    crossed_border=state.has_crossed_border,
                    sensor_data=sensor_data,
                    confidence=float(confidence),
                )

                border_distance = round(line_distance(center, camera.border_line), 2)
                log_event(
                    camera_id=camera.camera_id,
                    track_id=track_id,
                    object_name="Intruder",
                    confidence=float(confidence),
                    risk=risk_level,
                    risk_score=risk_score,
                    sensor_data=sensor_data,
                    behavior=behavior,
                    crossed_border=state.has_crossed_border,
                    group_size=group_size,
                    speed=round(speed, 2),
                    border_distance=border_distance,
                )

                if should_trigger_alert(state, risk_level, state.has_crossed_border):
                    snapshot_path = os.path.join(
                        "snapshots",
                        f"{camera.camera_id}_track_{track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    )
                    os.makedirs("snapshots", exist_ok=True)
                    cv2.imwrite(snapshot_path, frame)
                    send_alert_email(
                        camera_name=camera.name,
                        detected_object="Intruder",
                        risk_level=risk_level,
                        track_id=track_id,
                        behavior=behavior,
                        crossed_border=state.has_crossed_border,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        snapshot_path=snapshot_path,
                    )
                    state.last_alert_time = time.time()

                annotate_frame(
                    frame=frame,
                    box=(x1, y1, x2, y2),
                    track_id=track_id,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    behavior=behavior,
                    crossed_border=state.has_crossed_border,
                )

        metrics_text = (
            f"Motion:{sensor_data['motion']} Brightness:{sensor_data['brightness']} "
            f"IR:{sensor_data['infrared_proxy']}"
        )
        cv2.putText(frame, metrics_text, (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyWindow(window_name)


def main() -> None:
    initialize_logger()
    for camera in CAMERAS:
        process_camera(camera)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
