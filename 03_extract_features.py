import os
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---

# 1. Path to the extracted frames from Step 2.
FRAME_SOURCE_PATH = "data/frames"

# 2. Path where the extracted features will be saved.
FEATURE_DEST_PATH = "data/features"

# 3. Test set size (e.g., 0.2 means 20% of the data will be for testing).
TEST_SET_SIZE = 0.2

# 4. HOG parameters - a fixed size for all images is required.
IMAGE_RESIZE_DIM = (64, 128) # (width, height)


def get_all_frame_paths_and_labels():
    """Gathers all frame paths and their corresponding labels."""
    frame_paths = []
    labels = []
    
    class_folders = sorted(os.listdir(FRAME_SOURCE_PATH))
    
    for class_name in class_folders:
        class_path = os.path.join(FRAME_SOURCE_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for frame_filename in os.listdir(class_path):
            frame_paths.append(os.path.join(class_path, frame_filename))
            labels.append(class_name)
            
    return frame_paths, labels

def extract_features_for_paths(paths):
    """Extracts HOG features for a given list of image paths."""
    features = []
    # tqdm adds a progress bar, which is helpful for long processes
    for path in tqdm(paths, desc="Extracting HOG Features"):
        try:
            # Read image
            image = cv2.imread(path)
            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Resize image to a fixed dimension
            resized_image = cv2.resize(gray_image, IMAGE_RESIZE_DIM)
            
            # Initialize HOG descriptor
            hog = cv2.HOGDescriptor()
            # Compute HOG features
            hog_features = hog.compute(resized_image)
            # Flatten the features to a 1D array
            features.append(hog_features.flatten())
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Append a zero vector if an image is corrupted
            # The size needs to match a valid HOG descriptor output
            features.append(np.zeros(3780)) # Size for a 64x128 image with default HOG params
            
    return np.array(features)


def main():
    """Main function to run the feature extraction process."""
    print("🚀 Starting feature extraction process...")
    os.makedirs(FEATURE_DEST_PATH, exist_ok=True)
    
    # 1. Get all frame paths and labels
    print("1. Gathering all frame paths...")
    all_paths, all_labels = get_all_frame_paths_and_labels()
    print(f"Found {len(all_paths)} frames across {len(set(all_labels))} classes.")
    
    # 2. Split data into training and testing sets
    print("2. Splitting data into training and testing sets...")
    # stratify=all_labels ensures the class distribution is the same in train and test sets
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        all_paths, all_labels, test_size=TEST_SET_SIZE, random_state=42, stratify=all_labels
    )
    print(f"Training set size: {len(X_train_paths)} frames")
    print(f"Testing set size: {len(X_test_paths)} frames")

    # 3. Extract features for the training set
    print("\n3. Processing TRAINING data...")
    X_train_features = extract_features_for_paths(X_train_paths)
    
    # 4. Extract features for the testing set
    print("\n4. Processing TESTING data...")
    X_test_features = extract_features_for_paths(X_test_paths)
    
    # 5. Save all the processed data
    print("\n5. Saving features and labels to disk...")
    joblib.dump(X_train_features, os.path.join(FEATURE_DEST_PATH, "X_train.pkl"))
    joblib.dump(y_train, os.path.join(FEATURE_DEST_PATH, "y_train.pkl"))
    joblib.dump(X_test_features, os.path.join(FEATURE_DEST_PATH, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(FEATURE_DEST_PATH, "y_test.pkl"))
    
    print("\n🎉 Feature extraction complete!")
    print(f"Data saved in: {os.path.abspath(FEATURE_DEST_PATH)}")

if __name__ == "__main__":
    main()