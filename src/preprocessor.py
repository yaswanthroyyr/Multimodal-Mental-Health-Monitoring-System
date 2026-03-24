import cv2
import os

# --- FIX: explicit import path for MoviePy 2.0+ ---
from moviepy.video.io.VideoFileClip import VideoFileClip
# --------------------------------------------------

def extract_audio_from_video(video_path, output_audio_path):
    """Separates audio track from video file."""
    try:
        # Load the video clip
        my_clip = VideoFileClip(video_path)
        
        # Check if video has audio
        if my_clip.audio is None:
            print(f"⚠️ Warning: No audio found in {video_path}")
            my_clip.close()
            return

        # Extract audio
        my_clip.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
        my_clip.close() # Important to close the file to release memory
        print(f"🔊 Audio extracted to: {output_audio_path}")
        
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")

def extract_frames(video_path, output_folder, interval=1):
    """
    Extracts frames every 'interval' seconds.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 24 # Fallback
    frame_rate = int(fps * interval) 
    
    count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_rate == 0:
            frame_name = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
        count += 1
        
    cap.release()
    print(f"🖼️  Extracted {saved_count} frames to: {output_folder}")

if __name__ == "__main__":
    pass