import argparse
from src.config import VIDEOS
from src.pipeline import run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", choices=VIDEOS.keys(), required=True, help="Choose a video preset")
    ap.add_argument("--no_show", action="store_true", help="Don't show video window (faster)")
    args = ap.parse_args()

    cfg = VIDEOS[args.video]

    print(f"--- Starting Traffic Counter for {args.video.upper()} ---")

    run(
        video_name=args.video,
        input_path=cfg["input_path"],
        roi_path=cfg["roi_path"],
        out_video_path=cfg["out_video"],
        out_csv_path=cfg["out_csv"],
        show=(not args.no_show),
    )

if __name__ == "__main__":
    main()
