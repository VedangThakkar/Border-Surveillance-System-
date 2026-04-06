import mimetypes
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OWNER_EMAIL = os.getenv("OWNER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")
LIVE_FEED_LINK = os.getenv("LIVE_FEED_LINK", "Local camera feed")


def send_alert_email(
    camera_name: str,
    detected_object: str,
    risk_level: str,
    track_id: int,
    behavior: str,
    crossed_border: bool,
    timestamp: str,
    snapshot_path: str | None = None,
) -> bool:
    if not OWNER_EMAIL or not APP_PASSWORD or not TO_EMAIL:
        print("Alert skipped: email settings are incomplete in .env")
        return False

    subject = f"[{risk_level}] Border Alert - {camera_name} - Track {track_id}"
    crossing_text = "YES" if crossed_border else "NO"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = OWNER_EMAIL
    msg["To"] = TO_EMAIL
    msg.set_content(
        "\n".join(
            [
                "Border Surveillance Alert",
                "",
                f"Camera: {camera_name}",
                f"Object: {detected_object}",
                f"Track ID: {track_id}",
                f"Behavior: {behavior}",
                f"Risk Level: {risk_level}",
                f"Border Crossed: {crossing_text}",
                f"Timestamp: {timestamp}",
                f"Live Feed: {LIVE_FEED_LINK}",
            ]
        )
    )

    if snapshot_path and Path(snapshot_path).exists():
        mime_type, _ = mimetypes.guess_type(snapshot_path)
        maintype, subtype = (mime_type or "image/jpeg").split("/", 1)
        with open(snapshot_path, "rb") as image_file:
            msg.add_attachment(
                image_file.read(),
                maintype=maintype,
                subtype=subtype,
                filename=Path(snapshot_path).name,
            )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(OWNER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print(f"Alert email sent for track {track_id}")
        return True
    except Exception as exc:
        print(f"Email failed: {exc}")
        return False
