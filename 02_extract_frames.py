import os
import cv2 # OpenCV library for video processing
import math

# --- Configuration ---

# 1. Path to the filtered 10-class dataset from Step 1.
VIDEO_SOURCE_PATH = "data/processed/UCF-10"

# 2. Path where the extracted frames will be saved.
FRAME_DEST_PATH = "data/frames"

# 3. Number of frames to extract per second of video.
#    A value of 1 means we grab one frame every second.
FRAMES_PER_SECOND = 1

def extract_frames():
    """
    Iterates through all videos in the source path, extracts frames at a
    specified rate, and saves them to the destination path.
    """
    print("🚀 Starting frame extraction process...")
    
    if not os.path.exists(VIDEO_SOURCE_PATH):
        print(f"❌ Error: The video source path does not exist: {VIDEO_SOURCE_PATH}")
        return
        
    os.makedirs(FRAME_DEST_PATH, exist_ok=True)

    # Loop over each class folder ('WalkingWithDog', 'Typing', etc.)
    for class_name in os.listdir(VIDEO_SOURCE_PATH):
        class_path = os.path.join(VIDEO_SOURCE_PATH, class_name)
        dest_class_path = os.path.join(FRAME_DEST_PATH, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        if not os.path.isdir(class_path):
            continue
        
        print(f"\n  Extracting frames for class: '{class_name}'")

        # Loop over each video in the class folder
        for video_filename in os.listdir(class_path):
            video_path = os.path.join(class_path, video_filename)
            
            try:
                # Use OpenCV to open the video file
                video_capture = cv2.VideoCapture(video_path)
                
                # Get the video's frame rate (fps)
                fps = video_capture.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    print(f"    ⚠️ Warning: Could not get FPS for {video_filename}. Using default 30.")
                    fps = 30 # Use a default if fps is not available
                
                # Calculate the interval at which to capture frames
                frame_capture_interval = int(fps / FRAMES_PER_SECOND)

                frame_count = 0
                saved_frame_count = 0
                while video_capture.isOpened():
                    # Read the next frame
                    ret, frame = video_capture.read()
                    
                    if not ret:
                        break # End of video
                    
                    # Check if this frame is one we want to save
                    if frame_count % frame_capture_interval == 0:
                        # Construct the output filename
                        video_name_without_ext = os.path.splitext(video_filename)[0]
                        frame_filename = f"{video_name_without_ext}_frame_{saved_frame_count}.jpg"
                        frame_save_path = os.path.join(dest_class_path, frame_filename)
                        
                        # Save the frame as a JPG image
                        cv2.imwrite(frame_save_path, frame)
                        saved_frame_count += 1
                        
                    frame_count += 1
                
                video_capture.release()
            except Exception as e:
                print(f"    ❌ Error processing {video_filename}: {e}")

    print("\n🎉 Frame extraction complete! Your frames are saved at:", FRAME_DEST_PATH)

if __name__ == "__main__":
    extract_frames()