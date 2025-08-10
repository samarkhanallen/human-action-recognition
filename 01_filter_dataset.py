import os
import shutil

# --- Configuration ---

# 1. Paste the FULL, absolute path to your UCF-101 folder.
#    This is the folder that contains 'train', 'test', and 'val'.
#    Example: r"C:\Users\asus\Desktop\human-action-recognition\data\raw\UCF-101"
BASE_INPUT_PATH = r"C:\Users\asus\Desktop\human-action-recognition\data\raw\UCF-101"

# 2. Path where the smaller, 10-class dataset will be saved.
BASE_OUTPUT_PATH = "data/processed/UCF-10"

# 3. The dataset splits we need to search through.
SPLITS = ["train", "test", "val"]

# 4. The 10 action classes we want to collect.
SELECTED_CLASSES = [
    "WalkingWithDog", "Running", "Skipping", "JumpRope",
    "ApplyLipstick", "BrushingTeeth", "Typing", "WritingOnBoard",
    "PlayingGuitar", "PlayingPiano"
]


def filter_and_combine_dataset():
    """
    Searches through train, test, and val folders, finds all videos
    for the selected classes, and copies them into a single destination folder per class.
    """
    print("🚀 Starting final dataset filtering process...")
    
    if "PASTE" in BASE_INPUT_PATH:
        print("❌ Error: Please update the 'BASE_INPUT_PATH' variable in the script.")
        return

    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    total_files_copied = 0

    # Loop through each class we want
    for class_name in SELECTED_CLASSES:
        print(f"\nProcessing class: '{class_name}'")
        dest_class_dir = os.path.join(BASE_OUTPUT_PATH, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        class_files_copied = 0

        # Loop through each split folder to find videos for this class
        for split in SPLITS:
            src_class_dir = os.path.join(BASE_INPUT_PATH, split, class_name)

            if os.path.exists(src_class_dir):
                files_to_copy = os.listdir(src_class_dir)
                for file_name in files_to_copy:
                    # Add split name to filename to avoid overwrites if names are identical
                    new_file_name = f"{split}_{file_name}"
                    src_file_path = os.path.join(src_class_dir, file_name)
                    dest_file_path = os.path.join(dest_class_dir, new_file_name)
                    shutil.copy2(src_file_path, dest_file_path)
                
                class_files_copied += len(files_to_copy)

        if class_files_copied > 0:
            print(f"  ✅ Found and copied {class_files_copied} total files for '{class_name}'.")
            total_files_copied += class_files_copied
        else:
            print(f"  ⚠️ Warning: No files found for '{class_name}' in any split.")


    print(f"\n🎉 Filtering complete! Copied a total of {total_files_copied} files.")
    print(f"Your combined dataset is ready at: {os.path.abspath(BASE_OUTPUT_PATH)}")

if __name__ == "__main__":
    filter_and_combine_dataset()