import os
import cv2
import joblib
import numpy as np
import argparse
from collections import defaultdict

# --- Configuration & Model Loading ---
MODEL_PATH = "models"
VOCAB_PATH = "data/vocabulary"
IMAGE_RESIZE_DIM = (64, 128)
FRAMES_PER_SECOND = 1

print("Loading models...")
try:
    # Load the models we saved in the previous steps
    kmeans = joblib.load(os.path.join(VOCAB_PATH, "vocabulary.pkl"))
    svm = joblib.load(os.path.join(MODEL_PATH, "svm_video_model.pkl"))
    le = joblib.load(os.path.join(MODEL_PATH, "label_encoder_video.pkl"))
    num_clusters = kmeans.n_clusters
    hog = cv2.HOGDescriptor()
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please make sure you have run the training scripts successfully.")
    exit()

def predict_action(video_path):
    """
    Extracts features from a video, builds a histogram, and predicts the action.
    """
    # 1. Extract frames from the video
    video_features = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30 # Default fps
        
        frame_interval = int(fps / FRAMES_PER_SECOND)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % frame_interval == 0:
                # 2. Extract HOG for each frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, IMAGE_RESIZE_DIM)
                hog_features = hog.compute(resized)
                video_features.append(hog_features.flatten())

            frame_count += 1
        cap.release()
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

    if not video_features:
        print("Could not extract any features from the video.")
        return None

    # 3. Predict visual words for the video's frames
    visual_words = kmeans.predict(np.array(video_features))

    # 4. Create a normalized histogram
    histogram, _ = np.histogram(visual_words, bins=np.arange(num_clusters + 1))
    if np.sum(histogram) > 0:
        histogram = histogram.astype(float)
        histogram /= np.linalg.norm(histogram)

    # 5. Reshape histogram for SVM prediction and predict
    # The SVM model expects a 2D array, so we reshape our single histogram
    prediction_encoded = svm.predict(histogram.reshape(1, -1))
    
    # 6. Decode the prediction to its original label
    prediction_label = le.inverse_transform(prediction_encoded)
    
    return prediction_label[0]


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Predict human action from a video file.")
    parser.add_argument("--video", required=True, help="Path to the video file for prediction.")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at '{args.video}'")
    else:
        print(f"Analyzing video: {args.video}")
        predicted_action = predict_action(args.video)
        if predicted_action:
            # The label might look like 'v_PlayingGuitar', so we remove the 'v_' prefix
            clean_action = predicted_action.replace('v_', '')
            print("\n" + "="*30)
            print(f"✅ Predicted Action: {clean_action}")
            print("="*30)