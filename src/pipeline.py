import json
import os
import math
import cv2
from ultralytics import YOLO

from .geometry import cross_sign
from .logger import EventLogger
from .config import (
    WEIGHTS,
    CLASS_FILTER,
    TRACKER_CFG,
    RED_RESTRICTED_CLASS_IDS,
    RED_SEGMENT_MARGIN_PX,
    BLUE_RESTRICTED_CLASS_IDS,
    BLUE_DEDUP_GAP_SECONDS,
    BLUE_DEDUP_MIN_IOU,
    BLUE_DEDUP_MAX_CENTER_DIST_PX,
    BLUE_SEGMENT_MARGIN_PX,
    LINE_DEADZONE_PX,
    LINE_STABLE_FRAMES,
    LINE_REARM_PX,
)

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

_ALLOWED_DIRECTION_RULES = {"neg_to_pos_is_in", "pos_to_neg_is_in"}
_ALLOWED_DIRECTIONS = {"IN", "OUT"}
_ALLOWED_VIOLATION_DIRECTIONS = {"ANY", "IN", "OUT"}

_DEFAULT_VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
_DEFAULT_RED_RESTRICTED_CLASS_IDS = {1, 3}
_DEFAULT_BLUE_RESTRICTED_CLASS_IDS = {2, 5, 7}


def _direction_from_sides(prev, curr, direction_rule: str) -> str:
    if direction_rule == "neg_to_pos_is_in":
        return "IN" if (prev < 0 and curr > 0) else "OUT"
    return "IN" if (prev > 0 and curr < 0) else "OUT"


def _draw_bottom_left_violation_text(img, total_violations: int):
    h, _ = img.shape[:2]
    text = f"VIOLATIONS: {total_violations}"
    x, y = 20, h - 30
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

    cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), BLACK, -1)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2, cv2.LINE_AA)
    return img


def _draw_line_label(img, pts, label: str, color, y_offset: int = -10):
    if not label:
        return img
    x = int((pts[0][0] + pts[1][0]) / 2)
    y = int((pts[0][1] + pts[1][1]) / 2) + int(y_offset)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x0 = max(10, min(x - tw // 2, img.shape[1] - tw - 10))
    y0 = max(th + 10, min(y, img.shape[0] - 10))

    cv2.rectangle(img, (x0 - 6, y0 - th - 6), (x0 + tw + 6, y0 + 6), BLACK, -1)
    cv2.putText(img, label, (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def _require_line(roi: dict, key: str) -> dict:
    if key not in roi:
        raise KeyError(f"ROI missing required key: '{key}'")
    line = roi[key]
    if not isinstance(line, dict):
        raise TypeError(f"ROI['{key}'] must be an object/dict")
    return line


def _require_pts(line: dict, key: str) -> list:
    pts = line.get("pts", None)
    if not isinstance(pts, list) or len(pts) != 2:
        raise ValueError(f"ROI['{key}']['pts'] must be a list of exactly 2 points")
    for i, p in enumerate(pts):
        if (not isinstance(p, (list, tuple))) or len(p) != 2:
            raise ValueError(f"ROI['{key}']['pts'][{i}] must be [x, y]")
    return pts


def _normalize_rule(rule: str, key: str) -> str:
    if not isinstance(rule, str):
        raise TypeError(f"ROI['{key}']['direction_rule'] must be a string")
    if rule not in _ALLOWED_DIRECTION_RULES:
        raise ValueError(
            f"ROI['{key}']['direction_rule']='{rule}' is invalid. "
            f"Allowed: {sorted(_ALLOWED_DIRECTION_RULES)}"
        )
    return rule


def _normalize_direction(value: str, key: str, field: str, allowed: set) -> str:
    if value is None:
        raise ValueError(f"ROI['{key}']['{field}'] is required")
    if not isinstance(value, str):
        raise TypeError(f"ROI['{key}']['{field}'] must be a string")
    v = value.strip().upper()
    if v not in allowed:
        raise ValueError(
            f"ROI['{key}']['{field}']='{value}' is invalid. Allowed: {sorted(allowed)}"
        )
    return v


def _normalize_class_id_list(value, field_name: str):
    if value is None:
        return None, set(_DEFAULT_VEHICLE_CLASS_IDS)

    if not isinstance(value, (list, tuple, set)):
        raise TypeError(f"{field_name} must be a list/tuple/set of ints, got {type(value)}")

    value = list(value)
    if len(value) == 0:
        return None, set(_DEFAULT_VEHICLE_CLASS_IDS)

    yolo_classes = [int(x) for x in value]
    return yolo_classes, set(yolo_classes)


def _signed_distance_px(p, a, b) -> float:
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return float(cross_sign(p, a, b)) / (length + 1e-9)


def _side_from_dist(dist_px: float, deadzone_px: float) -> int:
    if abs(dist_px) < deadzone_px:
        return 0
    return 1 if dist_px > 0 else -1


def _update_line_cross_debounced(track_id: int, px: float, py: float, pts, state_dict: dict,
                                 deadzone_px: float, stable_frames: int, rearm_px: float):
    stable_frames = max(1, int(stable_frames))
    deadzone_px = max(0.0, float(deadzone_px))
    rearm_px = max(0.0, float(rearm_px))

    a, b = pts[0], pts[1]
    dist_px = _signed_distance_px((px, py), a, b)
    side = _side_from_dist(dist_px, deadzone_px)

    st = state_dict.get(track_id)
    if st is None:
        st = {
            "confirmed": None,
            "observed": 0,
            "stable": 0,
            "rearmed": True,
        }
        state_dict[track_id] = st

    if (not st["rearmed"]) and (abs(dist_px) >= rearm_px):
        st["rearmed"] = True

    if side == 0:
        st["observed"] = 0
        st["stable"] = 0
        return False, None, None, dist_px

    if st["confirmed"] is None:
        st["confirmed"] = side
        st["observed"] = side
        st["stable"] = stable_frames
        return False, None, None, dist_px

    if st["observed"] == side:
        st["stable"] += 1
    else:
        st["observed"] = side
        st["stable"] = 1

    if side != st["confirmed"] and st["stable"] >= stable_frames and st["rearmed"]:
        prev = st["confirmed"]
        curr = side
        st["confirmed"] = side
        st["rearmed"] = False
        return True, prev, curr, dist_px

    return False, None, None, dist_px


def _blue_point_within_segment_window(px: float, py: float, pts, margin_px: float) -> bool:
    """
    Returns True if point projects onto the finite segment span (plus margin).
    This prevents false crossings from the infinite extension of the line.
    """
    ax, ay = float(pts[0][0]), float(pts[0][1])
    bx, by = float(pts[1][0]), float(pts[1][1])
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-6:
        return True

    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    seg_len = math.sqrt(ab2)
    pad = float(margin_px) / max(seg_len, 1e-6)
    return (-pad) <= t <= (1.0 + pad)


def _update_blue_cross_segmented(track_id: int, px: float, py: float, pts, state_dict: dict,
                                 deadzone_px: float, stable_frames: int, rearm_px: float, margin_px: float):
    """
    BLUE-only wrapper:
      - If point is outside the segment window, we IGNORE and reset that track's BLUE state.
      - Otherwise, use the standard debounced crossing logic.
    """
    if not _blue_point_within_segment_window(px, py, pts, margin_px):
        if track_id in state_dict:
            del state_dict[track_id]
        return False, None, None, None

    return _update_line_cross_debounced(
        track_id, px, py, pts, state_dict,
        deadzone_px=deadzone_px,
        stable_frames=stable_frames,
        rearm_px=rearm_px,
    )


def _parse_blue_line_c(line_c: dict):
    allowed = {"name", "color", "pts", "direction_rule"}
    extra = set(line_c.keys()) - allowed
    if extra:
        raise ValueError(f"ROI['line_c'] has unused fields: {sorted(extra)}. Allowed: {sorted(allowed)}")

    name = line_c.get("name", None)
    if (not isinstance(name, str)) or (not name.strip()):
        raise ValueError("ROI['line_c']['name'] must be a non-empty string")

    color = line_c.get("color", None)
    if (not isinstance(color, str)) or (color.strip().lower() != "blue"):
        raise ValueError("ROI['line_c']['color'] must be exactly 'blue'")

    pts = _require_pts(line_c, "line_c")
    rule = _normalize_rule(line_c.get("direction_rule", "neg_to_pos_is_in"), "line_c")

    return name.strip(), pts, rule


def _bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 1e-9:
        return 0.0
    return inter / denom


class _BlueViolationDeduper:
    def __init__(self, max_gap_frames: int, min_iou: float, max_center_dist_px: float):
        self.max_gap_frames = max(1, int(max_gap_frames))
        self.min_iou = float(min_iou)
        self.max_center_dist_px = float(max_center_dist_px)
        self.seen_track_ids = set()
        self.events = []

    @staticmethod
    def _size_similar(b1, b2) -> bool:
        w1 = max(1.0, b1[2] - b1[0])
        h1 = max(1.0, b1[3] - b1[1])
        w2 = max(1.0, b2[2] - b2[0])
        h2 = max(1.0, b2[3] - b2[1])
        wr = w1 / w2
        hr = h1 / h2
        return (0.6 <= wr <= 1.4) and (0.6 <= hr <= 1.4)

    def should_log(self, track_id: int, frame_idx: int, bbox_xyxy, cx: float, cy: float) -> bool:
        if track_id in self.seen_track_ids:
            return False

        for ev in self.events:
            dt = frame_idx - ev["frame"]
            if dt < 0 or dt > self.max_gap_frames:
                continue

            prev_bbox = ev["bbox"]
            if not self._size_similar(prev_bbox, bbox_xyxy):
                continue

            iou = _bbox_iou_xyxy(prev_bbox, bbox_xyxy)
            if iou >= self.min_iou:
                self.seen_track_ids.add(track_id)
                return False

            dx = float(cx - ev["cx"])
            dy = float(cy - ev["cy"])
            if (dx * dx + dy * dy) <= (self.max_center_dist_px * self.max_center_dist_px):
                self.seen_track_ids.add(track_id)
                return False

        self.seen_track_ids.add(track_id)
        self.events.append({"frame": int(frame_idx), "bbox": tuple(bbox_xyxy), "cx": float(cx), "cy": float(cy)})
        return True


def run(video_name, input_path, roi_path, out_video_path, out_csv_path, show=True):
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    with open(roi_path, "r", encoding="utf-8") as f:
        roi = json.load(f)

    warning_seconds = roi.get("warning_seconds", 0.0)
    try:
        warning_seconds = float(warning_seconds)
    except Exception as e:
        raise ValueError(f"ROI['warning_seconds'] must be a number, got {warning_seconds!r}") from e
    if warning_seconds < 0:
        warning_seconds = 0.0

    red_enabled = False
    red_pts = None
    red_rule = None
    red_label = None
    red_violation_direction = None
    
    if "line_a" in roi:
        line_a = roi["line_a"]
        if isinstance(line_a, dict):
            red_pts = _require_pts(line_a, "line_a")
            red_rule = _normalize_rule(line_a.get("direction_rule", "neg_to_pos_is_in"), "line_a")
            red_label = str(line_a.get("name", "LINE_A"))
            red_violation_direction = _normalize_direction(
                line_a.get("violation_direction", "ANY"),
                "line_a",
                "violation_direction",
                _ALLOWED_VIOLATION_DIRECTIONS,
            )
            red_enabled = True

    green_enabled = False
    green_pts = None
    green_rule = None
    green_label = None
    green_count_direction = None

    if "line_b" in roi:
        line_b = roi["line_b"]
        if isinstance(line_b, dict):
            green_pts = _require_pts(line_b, "line_b")
            green_rule = _normalize_rule(line_b.get("direction_rule", "neg_to_pos_is_in"), "line_b")
            green_label = str(line_b.get("name", "LINE_B"))
            green_count_direction = _normalize_direction(
                line_b.get("count_direction", "IN"),
                "line_b",
                "count_direction",
                _ALLOWED_DIRECTIONS,
            )
            green_enabled = True

    blue_enabled = False
    blue_label = None
    blue_pts = None
    blue_rule = None
    if "line_c" in roi:
        line_c = roi["line_c"]
        if isinstance(line_c, dict):
            blue_label, blue_pts, blue_rule = _parse_blue_line_c(line_c)
            blue_enabled = True

    yolo_classes, vehicle_class_ids = _normalize_class_id_list(CLASS_FILTER, "CLASS_FILTER")

    if RED_RESTRICTED_CLASS_IDS is None:
        red_restricted_ids = set(_DEFAULT_RED_RESTRICTED_CLASS_IDS)
    else:
        if isinstance(RED_RESTRICTED_CLASS_IDS, (list, tuple, set)):
            red_restricted_ids = {int(x) for x in RED_RESTRICTED_CLASS_IDS}
        else:
            raise TypeError("RED_RESTRICTED_CLASS_IDS must be a list/tuple/set of ints or None")

    if BLUE_RESTRICTED_CLASS_IDS is None:
        blue_restricted_ids = set(_DEFAULT_BLUE_RESTRICTED_CLASS_IDS)
    else:
        if isinstance(BLUE_RESTRICTED_CLASS_IDS, (list, tuple, set)):
            blue_restricted_ids = {int(x) for x in BLUE_RESTRICTED_CLASS_IDS}
        else:
            raise TypeError("BLUE_RESTRICTED_CLASS_IDS must be a list/tuple/set of ints or None")

    model = YOLO(WEIGHTS)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    warn_frames = int(round(warning_seconds * fps)) if warning_seconds > 0 else 0

    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    logger = EventLogger(out_csv_path)

    cross_state_red = {} if red_enabled else None
    cross_state_green = {} if green_enabled else None
    cross_state_blue = {} if blue_enabled else None
    
    red_violation_ids = set()
    green_wrongway_ids = set()
    blue_violation_ids = set()
    
    green_count_ids = set()

    blue_violation_count = 0
    blue_deduper = None
    if blue_enabled:
        max_gap_frames = int(round(float(BLUE_DEDUP_GAP_SECONDS) * float(fps)))
        blue_deduper = _BlueViolationDeduper(
            max_gap_frames=max_gap_frames,
            min_iou=float(BLUE_DEDUP_MIN_IOU),
            max_center_dist_px=float(BLUE_DEDUP_MAX_CENTER_DIST_PX),
        )

    violation_highlight_until = {}
    standard_count = 0
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            show=False,
            verbose=False,
            classes=yolo_classes,
            conf=0.15,
            iou=0.5,
            imgsz=1280,
            tracker=TRACKER_CFG
        )

        vis = frame.copy()

        if red_enabled:
            cv2.line(vis, tuple(red_pts[0]), tuple(red_pts[1]), RED, 2)
            _draw_line_label(vis, red_pts, red_label, RED, y_offset=20)
        
        if green_enabled:
            cv2.line(vis, tuple(green_pts[0]), tuple(green_pts[1]), GREEN, 2)
            _draw_line_label(vis, green_pts, green_label, GREEN, y_offset=-12)

        if blue_enabled:
            cv2.line(vis, tuple(blue_pts[0]), tuple(blue_pts[1]), BLUE, 2)
            _draw_line_label(vis, blue_pts, blue_label, BLUE, y_offset=-35)

        if warn_frames > 0 and violation_highlight_until:
            expired = [tid for tid, until in violation_highlight_until.items() if frame_idx > until]
            for tid in expired:
                del violation_highlight_until[tid]

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().tolist()
            clss = boxes.cls.cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()

            for tid, cls_id, (x1, y1, x2, y2) in zip(ids, clss, xyxy):
                tid = int(tid)
                cls_id = int(cls_id)

                if cls_id not in vehicle_class_ids:
                    continue

                cls_name = model.names.get(cls_id, str(cls_id))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                px, py = cx, y2

                is_violator = (tid in red_violation_ids) or \
                              (tid in green_wrongway_ids) or \
                              (tid in blue_violation_ids)

                if red_enabled:
                    did_cross_red = False
                    prev_r = curr_r = None
                    
                    if _blue_point_within_segment_window(px, py, red_pts, RED_SEGMENT_MARGIN_PX):
                        did_cross_red, prev_r, curr_r, _ = _update_line_cross_debounced(
                            tid, px, py, red_pts, cross_state_red,
                            deadzone_px=LINE_DEADZONE_PX,
                            stable_frames=LINE_STABLE_FRAMES,
                            rearm_px=LINE_REARM_PX,
                        )
                    else:
                        if tid in cross_state_red:
                            del cross_state_red[tid]

                    if did_cross_red and (cls_id in red_restricted_ids) and tid not in red_violation_ids:
                        dir_red = _direction_from_sides(prev_r, curr_r, red_rule)
                        if red_violation_direction == "ANY" or dir_red == red_violation_direction:
                            red_violation_ids.add(tid)
                            
                            if tid in green_count_ids:
                                standard_count -= 1
                                green_count_ids.remove(tid)
                            
                            if warn_frames > 0:
                                violation_highlight_until[tid] = frame_idx + warn_frames

                            logger.log(
                                video_name, frame_idx, fps, tid, cls_name,
                                event="VIOLATION", line_name="RED_RESTRICTED",
                                direction=dir_red, cx=cx, cy=cy
                            )
                            is_violator = True

                if blue_enabled and (cls_id in blue_restricted_ids):
                    did_cross_blue, prev_b, curr_b, _ = _update_blue_cross_segmented(
                        tid, px, py, blue_pts, cross_state_blue,
                        deadzone_px=LINE_DEADZONE_PX,
                        stable_frames=LINE_STABLE_FRAMES,
                        rearm_px=LINE_REARM_PX,
                        margin_px=BLUE_SEGMENT_MARGIN_PX,
                    )
                    if did_cross_blue:
                        dir_blue = _direction_from_sides(prev_b, curr_b, blue_rule)
                        
                        if blue_deduper is not None and blue_deduper.should_log(
                            track_id=tid, frame_idx=frame_idx, bbox_xyxy=(x1, y1, x2, y2), cx=cx, cy=cy
                        ):
                            blue_violation_count += 1
                            blue_violation_ids.add(tid)
                            
                            if tid in green_count_ids:
                                standard_count -= 1
                                green_count_ids.remove(tid)

                            if warn_frames > 0:
                                violation_highlight_until[tid] = frame_idx + warn_frames

                            logger.log(
                                video_name, frame_idx, fps, tid, cls_name,
                                event="VIOLATION", line_name=blue_label,
                                direction=dir_blue, cx=cx, cy=cy
                            )
                            is_violator = True

                if green_enabled:
                    did_cross_green, prev_g, curr_g, _ = _update_line_cross_debounced(
                        tid, px, py, green_pts, cross_state_green,
                        deadzone_px=LINE_DEADZONE_PX,
                        stable_frames=LINE_STABLE_FRAMES,
                        rearm_px=LINE_REARM_PX,
                    )
                    if did_cross_green:
                        dir_green = _direction_from_sides(prev_g, curr_g, green_rule)

                        if dir_green == green_count_direction:
                            if tid not in green_count_ids and not is_violator:
                                green_count_ids.add(tid)
                                standard_count += 1
                                logger.log(
                                    video_name, frame_idx, fps, tid, cls_name,
                                    event="COUNT", line_name="GREEN_FLOW",
                                    direction=dir_green, cx=cx, cy=cy
                                )
                        else:
                            if tid not in green_wrongway_ids:
                                green_wrongway_ids.add(tid)
                                
                                if tid in green_count_ids:
                                    standard_count -= 1
                                    green_count_ids.remove(tid)
                                
                                if warn_frames > 0:
                                    violation_highlight_until[tid] = frame_idx + warn_frames
                                logger.log(
                                    video_name, frame_idx, fps, tid, cls_name,
                                    event="VIOLATION", line_name="GREEN_WRONG_WAY",
                                    direction=dir_green, cx=cx, cy=cy
                                )

                until = violation_highlight_until.get(tid, None)
                if until is not None and frame_idx <= until:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), RED, 2)
                    cv2.putText(
                        vis,
                        f"VIOLATION {cls_name} ID:{tid}",
                        (int(x1), max(20, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        RED,
                        2,
                        cv2.LINE_AA
                    )

        total_violations = len(red_violation_ids) + len(green_wrongway_ids) + int(blue_violation_count)

        vis = _draw_bottom_left_violation_text(vis, total_violations)
        cv2.putText(vis, f"STANDARD COUNT: {standard_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)

        writer.write(vis)

        if show:
            display_h = 800
            scale = display_h / max(h, 1)
            display_w = int(w * scale)
            if display_w > 0:
                cv2.imshow("Traffic Counter (Dual-Layer)", cv2.resize(vis, (display_w, display_h)))
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

        frame_idx += 1

    logger.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Done.")