import os
import joblib
import numpy as np
from tqdm import tqdm

# --- Configuration ---

# 1. Path to the features from Step 3.
FEATURE_SOURCE_PATH = "data/features"

# 2. Path to the vocabulary model from Step 4.
VOCAB_SOURCE_PATH = "data/vocabulary"

# 3. Path where the final histogram representations will be saved.
HISTOGRAM_DEST_PATH = "data/histograms"

def create_histograms():
    """
    Loads features and the vocabulary, then creates a histogram
    (a list of visual word IDs) for each frame.
    """
    print("🚀 Starting histogram creation process...")
    os.makedirs(HISTOGRAM_DEST_PATH, exist_ok=True)
    
    # 1. Load the vocabulary model
    print("1. Loading vocabulary model...")
    try:
        kmeans = joblib.load(os.path.join(VOCAB_SOURCE_PATH, "vocabulary.pkl"))
    except FileNotFoundError:
        print(f"❌ Error: Could not find 'vocabulary.pkl'. Please run Step 4.")
        return

    # 2. Load the feature and label files
    print("2. Loading feature sets and labels...")
    try:
        X_train_features = joblib.load(os.path.join(FEATURE_SOURCE_PATH, "X_train.pkl"))
        y_train = joblib.load(os.path.join(FEATURE_SOURCE_PATH, "y_train.pkl"))
        X_test_features = joblib.load(os.path.join(FEATURE_SOURCE_PATH, "X_test.pkl"))
        y_test = joblib.load(os.path.join(FEATURE_SOURCE_PATH, "y_test.pkl"))
    except FileNotFoundError:
        print(f"❌ Error: Feature files not found. Please run Step 3.")
        return

    # 3. Create histograms for the training set
    print("3. Creating histograms for the training set...")
    # 'kmeans.predict' finds the closest cluster (visual word) for each feature vector.
    X_train_hist = kmeans.predict(X_train_features)

    # 4. Create histograms for the testing set
    print("4. Creating histograms for the testing set...")
    X_test_hist = kmeans.predict(X_test_features)
    
    # Reshape the data to be a 2D array for compatibility with classifiers
    X_train_hist = X_train_hist.reshape(-1, 1)
    X_test_hist = X_test_hist.reshape(-1, 1)

    # 5. Save the final datasets
    print("5. Saving histogram data...")
    joblib.dump(X_train_hist, os.path.join(HISTOGRAM_DEST_PATH, "X_train.pkl"))
    joblib.dump(y_train, os.path.join(HISTOGRAM_DEST_PATH, "y_train.pkl"))
    joblib.dump(X_test_hist, os.path.join(HISTOGRAM_DEST_PATH, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(HISTOGRAM_DEST_PATH, "y_test.pkl"))

    print("\n🎉 Histogram creation complete!")
    print(f"Final data is ready for training in: {os.path.abspath(HISTOGRAM_DEST_PATH)}")

if __name__ == "__main__":
    create_histograms()