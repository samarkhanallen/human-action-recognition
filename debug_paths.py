import os

# --- Configuration ---
# !! IMPORTANT: Make sure this path is IDENTICAL to the one in your
#    '01_filter_dataset.py' script.
BASE_INPUT_PATH = "data/raw/UCF-101"

def check_paths():
    """
    A simple diagnostic tool to verify that the dataset paths are correct.
    """
    print("--- 🕵️ Running Path Diagnostic ---")
    
    # 1. Check if the main dataset folder exists
    print(f"\n[1] Checking for base folder...\n    Path: '{BASE_INPUT_PATH}'")
    if not os.path.exists(BASE_INPUT_PATH):
        print("    ❌ FAILURE: This folder does not exist.")
        print("    👉 SOLUTION: Please check the 'BASE_INPUT_PATH' variable. It must point directly to your UCF-101 folder.")
        return
    print("    ✅ SUCCESS: Base folder found.")
    
    # 2. Check for the 'train' subfolder
    train_path = os.path.join(BASE_INPUT_PATH, 'train')
    print(f"\n[2] Checking for 'train' split folder...\n    Path: '{train_path}'")
    if not os.path.exists(train_path):
        print("    ❌ FAILURE: The 'train' folder is missing from your base folder.")
        print(f"    FYI: Your base folder contains these items: {os.listdir(BASE_INPUT_PATH)}")
        print("    👉 SOLUTION: Make sure your UCF-101 folder directly contains the 'train', 'val', and 'test' folders.")
        return
    print("    ✅ SUCCESS: 'train' folder found.")

    # 3. Check for a sample class folder, e.g., 'Running'
    sample_class = "Running" # We use one of our 10 classes as a test
    class_path = os.path.join(train_path, sample_class)
    print(f"\n[3] Checking for a sample class folder...\n    Path: '{class_path}'")
    if not os.path.exists(class_path):
        print(f"    ❌ FAILURE: The sample class folder '{sample_class}' is missing from your 'train' folder.")
        print(f"    FYI: Your 'train' folder contains these class folders: {os.listdir(train_path)[:5]}...") # Shows first 5
        print("    👉 SOLUTION: Verify that the class folder names on your computer match the names in the 'SELECTED_CLASSES' list in the script exactly (they are case-sensitive).")
        return
    print("    ✅ SUCCESS: Sample class folder found.")

    print("\n\n--- ✅ All checks passed! Your paths seem correct. ---")

if __name__ == "__main__":
    check_paths()
