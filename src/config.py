import os

# Chỉ giữ lại asia1 theo yêu cầu
VIDEOS = {
    "asia1": {
        "input_path": "input_video/asia1.mp4",
        "roi_path": "roi_configs/asia1.json",
        "out_video": "output_video/asia1_annotated.mp4",
        "out_csv": "output_data/asia1_events.csv",
    },
}

# Model Settings
WEIGHTS = "yolo11n.pt"
TRACKER_CFG = "botsort.yaml"

# COCO IDs (Ultralytics default):
#  1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

# Used for inference-time filtering
CLASS_FILTER = VEHICLE_CLASS_IDS

# Red restricted lane violations apply ONLY to these vehicle class IDs
RED_RESTRICTED_CLASS_IDS = [1, 3]

# Blue restricted lane violations apply ONLY to these vehicle class IDs
BLUE_RESTRICTED_CLASS_IDS = [2, 5, 7]

# Blue dedup robustness (ID switching)
BLUE_DEDUP_GAP_SECONDS = 3.0
BLUE_DEDUP_MIN_IOU = 0.25
BLUE_DEDUP_MAX_CENTER_DIST_PX = 120.0

# ✅ BLUE-only fix:
# Prevent "infinite line" false crossings by requiring the crossing point to be
# within the segment span (with a small margin in pixels along the segment).
BLUE_SEGMENT_MARGIN_PX = 60.0
RED_SEGMENT_MARGIN_PX = 60.0

# ------------------------------------------------------------
# Step 4: Line-crossing stability controls (global defaults)
# ------------------------------------------------------------

LINE_DEADZONE_PX = 6.0
LINE_STABLE_FRAMES = 1
LINE_REARM_PX = 25.0