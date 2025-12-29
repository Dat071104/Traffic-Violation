import sys
import os
from src.config import VIDEOS
from src.roi_tool import roi_selector
from src.pipeline import run as run_pipeline

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    while True:
        clear_screen()
        print("==========================================")
        print("   TRAFFIC VIOLATION & COUNTING SYSTEM    ")
        print("==========================================")

        video_keys = list(VIDEOS.keys())
        print("Available Video Presets:")
        for i, key in enumerate(video_keys):
            print(f"  {i+1}. {key.upper()}")
        print("  Q. Quit")
        
        choice = input("\nSelect a video (1-3) or Q: ").strip().lower()
        
        if choice == 'q':
            print("Exiting application.")
            break
            
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(video_keys):
                print("Invalid selection.")
                input("Press Enter to continue...")
                continue
        except ValueError:
            print("Invalid input.")
            input("Press Enter to continue...")
            continue
            
        selected_key = video_keys[idx]
        cfg = VIDEOS[selected_key]
        
        print(f"\n--- Selected: {selected_key.upper()} ---")
        print(f"Input: {cfg['input_path']}")
        print(f"Config: {cfg['roi_path']}")

        print("\n[ROI SETUP]")
        if os.path.exists(cfg['roi_path']):
            redraw = input("Config exists. Do you want to redraw/edit ROI? (y/N): ").strip().lower()
        else:
            print("Config not found. You MUST draw ROI.")
            redraw = 'y'
            
        if redraw == 'y':
            print("Opening ROI Tool... (Press ENTER to save, ESC to cancel)")
            saved = roi_selector(
                video_path=cfg['input_path'], 
                out_json_path=cfg['roi_path']
            )
            if not saved and not os.path.exists(cfg['roi_path']):
                print("ROI setup cancelled and no config exists. Cannot run.")
                input("Press Enter to return to menu...")
                continue

        print("\n[PROCESSING]")
        print("Starting traffic analysis pipeline...")
        print("Press 'ESC' in the video window to stop early.")
        
        try:
            run_pipeline(
                video_name=selected_key,
                input_path=cfg['input_path'],
                roi_path=cfg['roi_path'],
                out_video_path=cfg['out_video'],
                out_csv_path=cfg['out_csv'],
                show=True
            )
        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nProcessing finished.")
        input("Press Enter to return to main menu...")

if __name__ == "__main__":
    main()