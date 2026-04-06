import os

import matplotlib.pyplot as plt
import pandas as pd


def generate_analytics(log_file: str = "logs.csv") -> None:
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    df = pd.read_csv(log_file)
    if df.empty:
        print("No events available for analytics.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    os.makedirs("analytics_output", exist_ok=True)

    risk_counts = df["risk"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
    plt.figure(figsize=(6, 4))
    risk_counts.plot(kind="bar", color=["#43a047", "#fdd835", "#e53935"])
    plt.title("Risk Level Distribution")
    plt.xlabel("Risk")
    plt.ylabel("Events")
    plt.tight_layout()
    plt.savefig("analytics_output/risk_distribution.png")
    plt.close()

    behavior_counts = df["behavior"].value_counts()
    plt.figure(figsize=(6, 4))
    behavior_counts.plot(kind="bar", color="#1e88e5")
    plt.title("Behavior Frequency")
    plt.xlabel("Behavior")
    plt.ylabel("Events")
    plt.tight_layout()
    plt.savefig("analytics_output/behavior_frequency.png")
    plt.close()

    intrusions = df[df["crossed_border"] == True].copy()
    if not intrusions.empty:
        intrusions["hour"] = intrusions["timestamp"].dt.hour
        hourly_intrusions = intrusions.groupby("hour").size()

        plt.figure(figsize=(7, 4))
        hourly_intrusions.plot(kind="line", marker="o", color="#d81b60")
        plt.title("Border Crossings by Hour")
        plt.xlabel("Hour")
        plt.ylabel("Crossings")
        plt.tight_layout()
        plt.savefig("analytics_output/border_crossings_by_hour.png")
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.scatter(
            intrusions["border_distance"],
            intrusions["speed"],
            c=intrusions["risk_score"],
            cmap="inferno",
            alpha=0.7,
        )
        plt.title("Intrusion Risk Heatmap")
        plt.xlabel("Distance From Border")
        plt.ylabel("Speed")
        plt.colorbar(label="Risk Score")
        plt.tight_layout()
        plt.savefig("analytics_output/intrusion_heatmap.png")
        plt.close()

    print("Analytics generated in analytics_output/")


if __name__ == "__main__":
    generate_analytics()
