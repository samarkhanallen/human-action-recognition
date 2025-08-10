import os
import cv2
import joblib
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

# --- Configuration ---
FRAME_SOURCE_PATH = "data/frames"
VOCAB_SOURCE_PATH = "data/vocabulary"
VIDEO_HIST_DEST_PATH = "data/video_histograms" # New folder for our powerful features
TEST_SET_SIZE = 0.2
IMAGE_RESIZE_DIM = (64, 128)

def group_frames_by_video(frame_folders):
    """Groups frame paths by their original video name."""
    video_frames = defaultdict(list)
    for class_folder in frame_folders:
        class_path = os.path.join(FRAME_SOURCE_PATH, class_folder)
        if not os.path.isdir(class_path):
            continue
        for frame_name in os.listdir(class_path):
            # Assumes filename format is 'split_videoname_frame_num.jpg'
            video_id = "_".join(frame_name.split('_')[:-2])
            video_frames[video_id].append(os.path.join(class_path, frame_name))
    return video_frames

def create_video_level_histograms():
    """Creates a single histogram feature vector for each video."""
    print("🚀 Starting video-level histogram creation...")
    os.makedirs(VIDEO_HIST_DEST_PATH, exist_ok=True)
    
    # 1. Load the vocabulary
    print("1. Loading vocabulary...")
    try:
        kmeans = joblib.load(os.path.join(VOCAB_SOURCE_PATH, "vocabulary.pkl"))
        num_clusters = kmeans.n_clusters
    except FileNotFoundError:
        print("❌ Vocabulary not found. Please run Step 4 first.")
        return
        
    # 2. Group all frames by video
    print("2. Grouping frames by original video...")
    class_folders = sorted(os.listdir(FRAME_SOURCE_PATH))
    video_frames_map = group_frames_by_video(class_folders)
    print(f"Found {len(video_frames_map)} unique videos.")
    
    # 3. Create a histogram for each video
    X = [] # To store video histograms
    y = [] # To store video labels
    hog = cv2.HOGDescriptor()

    for video_id, frame_paths in tqdm(video_frames_map.items(), desc="Processing Videos"):
        video_features = []
        for frame_path in frame_paths:
            try:
                # Extract HOG for each frame in the video
                image = cv2.imread(frame_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(gray_image, IMAGE_RESIZE_DIM)
                hog_features = hog.compute(resized_image)
                video_features.append(hog_features.flatten())
            except Exception:
                continue # Skip corrupted frames
        
        if not video_features:
            continue

        # Predict the visual word for each frame's HOG feature
        visual_words = kmeans.predict(np.array(video_features))
        
        # Create a histogram of the visual words
        histogram, _ = np.histogram(visual_words, bins=np.arange(num_clusters + 1))
        
        # Normalize the histogram (L2 norm)
        if np.sum(histogram) > 0:
            histogram = histogram.astype(float)
            histogram /= np.linalg.norm(histogram)

        X.append(histogram)
        # The class name is the part of the video_id after the first underscore
        label = video_id.split('_', 1)[1].rsplit('_', 2)[0]
        y.append(label)

    # 4. Split and save the final dataset
    print("\n4. Splitting and saving the video-level dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=42, stratify=y)

    joblib.dump(X_train, os.path.join(VIDEO_HIST_DEST_PATH, "X_train.pkl"))
    joblib.dump(y_train, os.path.join(VIDEO_HIST_DEST_PATH, "y_train.pkl"))
    joblib.dump(X_test, os.path.join(VIDEO_HIST_DEST_PATH, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(VIDEO_HIST_DEST_PATH, "y_test.pkl"))
    
    print("\n🎉 Video-level feature creation complete!")

if __name__ == "__main__":
    create_video_level_histograms()