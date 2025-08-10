import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---

# 1. Path to the histograms from Step 5.
HISTOGRAM_SOURCE_PATH = "data/histograms"

# 2. Path where the final trained models will be saved.
MODEL_DEST_PATH = "models"


def train_and_evaluate():
    """
    Loads the final dataset, trains an SVM classifier, evaluates it,
    and saves the trained models.
    """
    print("🚀 Starting model training and evaluation process...")
    os.makedirs(MODEL_DEST_PATH, exist_ok=True)
    
    # 1. Load the dataset
    print("1. Loading final dataset...")
    try:
        X_train = joblib.load(os.path.join(HISTOGRAM_SOURCE_PATH, "X_train.pkl"))
        y_train = joblib.load(os.path.join(HISTOGRAM_SOURCE_PATH, "y_train.pkl"))
        X_test = joblib.load(os.path.join(HISTOGRAM_SOURCE_PATH, "X_test.pkl"))
        y_test = joblib.load(os.path.join(HISTOGRAM_SOURCE_PATH, "y_test.pkl"))
    except FileNotFoundError:
        print("❌ Error: Histogram files not found. Please run Step 5.")
        return

    # 2. Encode text labels into numbers
    print("2. Encoding labels...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # 3. Initialize and train the SVM Classifier
    print("3. Training the SVM classifier...")
    # A linear kernel is a good baseline. 'probability=True' is needed for some advanced uses.
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train_encoded)
    
    # 4. Make predictions on the test set
    print("4. Making predictions on the test set...")
    y_pred_encoded = model.predict(X_test)
    
    # 5. Evaluate the model
    print("\n--- Model Evaluation ---")
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
    print(f"✅ Overall Accuracy: {accuracy * 100:.2f}%")
    
    print("\n✅ Classification Report:")
    # Get class names from the label encoder to make the report readable
    class_names = le.classes_
    print(classification_report(y_test_encoded, y_pred_encoded, target_names=class_names))
    
    # 6. Save the trained model and the label encoder
    print("--- Saving Models ---")
    joblib.dump(model, os.path.join(MODEL_DEST_PATH, "svm_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DEST_PATH, "label_encoder.pkl"))
    
    print(f"✅ Models saved in: {os.path.abspath(MODEL_DEST_PATH)}")
    print("\n🎉 Project Complete!")


if __name__ == "__main__":
    train_and_evaluate()