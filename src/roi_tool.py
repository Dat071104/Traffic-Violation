import argparse
import json
import cv2
import numpy as np

PTS_A = []
PTS_B = []
PTS_C = []
ACTIVE = "A"
SCALE = 1.0


def on_mouse(event, x, y, flags, param):
    global SCALE, ACTIVE, PTS_A, PTS_B, PTS_C
    if event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(x * SCALE)
        real_y = int(y * SCALE)

        if ACTIVE == "A":
            if len(PTS_A) < 2:
                PTS_A.append([real_x, real_y])
        elif ACTIVE == "B":
            if len(PTS_B) < 2:
                PTS_B.append([real_x, real_y])
        else:
            if len(PTS_C) < 2:
                PTS_C.append([real_x, real_y])


def draw_line(vis_small, pts_big, color, thickness):
    if len(pts_big) == 2:
        p1 = (int(pts_big[0][0] / SCALE), int(pts_big[0][1] / SCALE))
        p2 = (int(pts_big[1][0] / SCALE), int(pts_big[1][1] / SCALE))
        cv2.line(vis_small, p1, p2, color, thickness)


def roi_selector(video_path, out_json_path, 
                 warning_seconds=2.0, 
                 line_a_rule="neg_to_pos_is_in", line_a_violation_dir="ANY",
                 line_b_rule="neg_to_pos_is_in", line_b_count_dir="IN",
                 line_c_rule="neg_to_pos_is_in", line_c_name="BLUE_RESTRICTED"):
    global SCALE, ACTIVE, PTS_A, PTS_B, PTS_C
    
    PTS_A = []
    PTS_B = []
    PTS_C = []
    ACTIVE = "A"

    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"ERROR: Cannot read video {video_path}")
        return

    target_height = 800
    h, w = frame.shape[:2]
    SCALE = h / target_height
    new_w = int(w / SCALE)
    new_h = int(h / SCALE)

    print(f"Original: {w}x{h}")
    print(f"Display:  {new_w}x{new_h} (Scale factor: {SCALE:.2f})")
    print("--- INSTRUCTIONS ---")
    print("1) Click 2 points for Line A (RED) — restricted lane (Optional)")
    print("2) Press '2' to switch to Line B (GREEN) — flow/wrong-way (Optional)")
    print("3) Press '3' to switch to Line C (BLUE) — restricted lane (Optional)")
    print("Keys: [1]=A, [2]=B, [3]=C, [R]=reset all, [ENTER]=save, [ESC]=quit without saving")

    cv2.namedWindow("ROI Selector")
    cv2.setMouseCallback("ROI Selector", on_mouse)

    saved = False

    while True:
        vis_small = cv2.resize(frame, (new_w, new_h))

        for p in PTS_A:
            cv2.circle(vis_small, (int(p[0] / SCALE), int(p[1] / SCALE)), 5, (0, 0, 255), -1)
        for p in PTS_B:
            cv2.circle(vis_small, (int(p[0] / SCALE), int(p[1] / SCALE)), 5, (0, 255, 0), -1)
        for p in PTS_C:
            cv2.circle(vis_small, (int(p[0] / SCALE), int(p[1] / SCALE)), 5, (255, 0, 0), -1)

        draw_line(vis_small, PTS_A, (0, 0, 255), 2)
        draw_line(vis_small, PTS_B, (0, 255, 0), 2)
        draw_line(vis_small, PTS_C, (255, 0, 0), 2)

        active_txt = "Line A (RED)" if ACTIVE == "A" else ("Line B (GREEN)" if ACTIVE == "B" else "Line C (BLUE)")
        cv2.putText(vis_small, f"ACTIVE: {active_txt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        info_txt = f"A: {len(PTS_A)}/2 | B: {len(PTS_B)}/2 | C: {len(PTS_C)}/2"
        cv2.putText(vis_small, info_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        cv2.imshow("ROI Selector", vis_small)

        k = cv2.waitKey(1) & 0xFF
        if k == 27: 
            print("Cancelled ROI selection.")
            break
        
        if k in (ord("r"), ord("R")):
            PTS_A.clear()
            PTS_B.clear()
            PTS_C.clear()
            ACTIVE = "A"
        if k == ord("1"):
            ACTIVE = "A"
        if k == ord("2"):
            ACTIVE = "B"
        if k == ord("3"):
            ACTIVE = "C"

        if k == 13: 
            if len(PTS_A) not in (0, 2):
                print("ERROR: Line A must have 0 or 2 points.")
                continue
            if len(PTS_B) not in (0, 2):
                print("ERROR: Line B must have 0 or 2 points.")
                continue
            if len(PTS_C) not in (0, 2):
                print("ERROR: Line C must have 0 or 2 points.")
                continue
            
            if len(PTS_A) == 0 and len(PTS_B) == 0 and len(PTS_C) == 0:
                print("ERROR: Please define at least one line (A, B, or C).")
                continue

            data = {
                "warning_seconds": float(warning_seconds)
            }

            if len(PTS_A) == 2:
                data["line_a"] = {
                    "name": "BANNED_LANE",
                    "pts": PTS_A,
                    "direction_rule": line_a_rule,
                    "violation_direction": line_a_violation_dir
                }

            if len(PTS_B) == 2:
                data["line_b"] = {
                    "name": "OPPOSING_FLOW",
                    "pts": PTS_B,
                    "direction_rule": line_b_rule,
                    "count_direction": line_b_count_dir
                }

            if len(PTS_C) == 2:
                data["line_c"] = {
                    "name": str(line_c_name),
                    "color": "blue",
                    "pts": PTS_C,
                    "direction_rule": line_c_rule
                }

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"Saved ROI configuration to: {out_json_path}")
            saved = True
            break

    cv2.destroyAllWindows()
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warning_seconds", type=float, default=2.0)
    args = ap.parse_args()

    roi_selector(args.video, args.out, warning_seconds=args.warning_seconds)