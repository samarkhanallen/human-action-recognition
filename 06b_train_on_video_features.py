import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
# Point this to our new video_histograms folder
HISTOGRAM_SOURCE_PATH = "data/video_histograms" 
MODEL_DEST_PATH = "models"

def train_and_evaluate_on_video_histograms():
    """Loads video-level histograms, trains, and evaluates the SVM."""
    print("🚀 Training and evaluating on POWERFUL video-level features...")
    os.makedirs(MODEL_DEST_PATH, exist_ok=True)
    
    # 1. Load the dataset
    print("1. Loading video-level dataset...")
    try:
        X_train = joblib.load(os.path.join(HISTOGRAM_SOURCE_PATH, "X_train.pkl"))
        y_train = joblib.load(os.path.join(HISTOGRAM_SOURCE_PATH, "y_train.pkl"))
        X_test = joblib.load(os.path.join(HISTOGRAM_SOURCE_PATH, "X_test.pkl"))
        y_test = joblib.load(os.path.join(HISTOGRAM_SOURCE_PATH, "y_test.pkl"))
    except FileNotFoundError:
        print("❌ Error: Histogram files not found. Please run the new Step 1 script first.")
        return

    # 2. Encode labels
    print("2. Encoding labels...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # 3. Train the SVM Classifier
    print("3. Training the SVM classifier...")
    # Using a non-linear kernel like 'rbf' can work better with histogram data
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    model.fit(X_train, y_train_encoded)
    
    # 4. Evaluate the model
    print("\n--- Model Evaluation ---")
    y_pred_encoded = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"✅ Overall Accuracy: {accuracy * 100:.2f}%")
    
    print("\n✅ Classification Report:")
    class_names = le.classes_
    print(classification_report(y_test_encoded, y_pred_encoded, target_names=class_names))
    
    # 5. Save the improved models
    print("--- Saving Improved Models ---")
    joblib.dump(model, os.path.join(MODEL_DEST_PATH, "svm_video_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DEST_PATH, "label_encoder_video.pkl"))
    
    print(f"✅ Improved models saved in: {os.path.abspath(MODEL_DEST_PATH)}")

if __name__ == "__main__":
    train_and_evaluate_on_video_histograms()