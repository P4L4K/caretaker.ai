import sys
import os

# Ensure backend directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import united_monitor

def main():
    print("=== VIDEO ELDERLY MONITOR ===")
    
    if len(sys.argv) < 2:
        print("Usage: python run_video_monitor.py <path_to_video_file>")
        print("Please provide a video file to analyze.")
        # Optional: Ask user for input if no arg provided
        video_path = input("Enter path to video file: ").strip()
        if not video_path:
            return
    else:
        video_path = sys.argv[1]

    # Smart path resolution
    if not os.path.exists(video_path):
        # 1. Try checking in the parent directory (project root)
        parent_path = os.path.join(os.path.dirname(current_dir), video_path)
        if os.path.exists(parent_path):
            video_path = parent_path
        else:
            # 2. Try handling "caretaker/" prefix if user included root folder name redundantly
            # e.g. "caretaker/fall_video/..." while inside "backend/"
            # We want to transform "caretaker/fall_video/..." to "../fall_video/..."
            if video_path.startswith("caretaker/") or video_path.startswith("caretaker\\"):
                # Remove "caretaker" and try looking in parent
                stripped_path = video_path[10:] # len("caretaker/")
                # Look in parent
                fixed_path = os.path.join(os.path.dirname(current_dir), stripped_path)
                # Look in current (unlikely but possible)
                if os.path.exists(fixed_path):
                     video_path = fixed_path
                elif os.path.exists(stripped_path):
                     video_path = stripped_path
            
            # 3. Simple parent check for straight filename or relative path
            # e.g. user typed "fall_video/..." but is in "backend/"
            simple_parent = os.path.join("..", video_path)
            if os.path.exists(simple_parent):
                video_path = simple_parent

    if not os.path.exists(video_path):
        print(f"Error: File not found: {video_path}")
        print(f"Current Directory: {os.getcwd()}")
        print("Tip: If the file is in 'fall_video', try using: ..\\fall_video\\your_video.mp4")
        return

    output_filename = "analyzed_" + os.path.basename(video_path)
    output_dir = os.path.join(os.path.dirname(current_dir), "processed_videos")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Analyzing video: {video_path}")
    print(f"Output will be saved to: {output_path}")

    # Simulate command line arguments for united_monitor
    sys.argv = [sys.argv[0], "--video", video_path, "--output", output_path]
    
    try:
        united_monitor.main()
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
