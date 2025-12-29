import csv
from datetime import timedelta


class EventLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.f = open(csv_path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "video", "time", "frame", "id", "class",
            "event", "line", "direction", "cx", "cy"
        ])

    def log(self, video_name, frame_idx, fps, track_id, cls_name,
            event, line_name, direction, cx, cy):
        t = str(timedelta(seconds=int(frame_idx / fps)))
        self.w.writerow([
            video_name, t, frame_idx, track_id, cls_name,
            event, line_name, direction, int(cx), int(cy)
        ])

    def close(self):
        self.f.close()
        print(f"[Logger] Data saved to {self.csv_path}")
    