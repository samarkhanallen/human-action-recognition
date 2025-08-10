import os
import joblib
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# --- Configuration ---

# 1. Path where features from Step 3 are stored.
FEATURE_SOURCE_PATH = "data/features"

# 2. Path where the new vocabulary model will be saved.
VOCAB_DEST_PATH = "data/vocabulary"

# 3. The number of "visual words" to create. This is the 'k' in K-Means.
#    A value between 200-500 is common for this kind of task.
NUM_CLUSTERS = 300


def build_vocabulary():
    """
    Uses K-Means clustering on the training features to build a visual vocabulary.
    """
    print("🚀 Starting vocabulary building process...")
    os.makedirs(VOCAB_DEST_PATH, exist_ok=True)
    
    # 1. Load the training features
    print("1. Loading training features...")
    try:
        X_train_features = joblib.load(os.path.join(FEATURE_SOURCE_PATH, "X_train.pkl"))
    except FileNotFoundError:
        print(f"❌ Error: Could not find 'X_train.pkl' in '{FEATURE_SOURCE_PATH}'.")
        print("👉 Please make sure you have successfully run Step 3.")
        return
        
    print(f"Loaded {X_train_features.shape[0]} feature vectors of size {X_train_features.shape[1]}.")

    # 2. Initialize and train the K-Means model
    print(f"2. Clustering features into {NUM_CLUSTERS} visual words using MiniBatchKMeans...")
    # MiniBatchKMeans is faster for large datasets.
    # We set a random_state to get reproducible results.
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=42, batch_size=2048)
    kmeans.fit(X_train_features)
    
    # 3. Save the trained K-Means model (our vocabulary)
    print("3. Saving the vocabulary model...")
    vocab_save_path = os.path.join(VOCAB_DEST_PATH, "vocabulary.pkl")
    joblib.dump(kmeans, vocab_save_path)
    
    print("\n🎉 Vocabulary building complete!")
    print(f"Vocabulary model saved at: {os.path.abspath(vocab_save_path)}")


if __name__ == "__main__":
    build_vocabulary()