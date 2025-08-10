# Human Action Recognition Pipeline

This repository contains a complete pipeline for human action recognition from video data. The system utilizes classical computer vision techniques, including Histogram of Oriented Gradients (HOG) for feature extraction and a Bag of Visual Words (BoVW) model constructed via K-Means clustering. A Support Vector Machine (SVM) is employed for the final classification task.

---

### ## Methodology and Results

The project evaluated two distinct feature representation strategies, yielding a significant performance differential.

1.  **Initial Frame-Level Model:** An initial model was developed to classify individual, independent frames. This methodology yielded a baseline accuracy of **22.31%**. Analysis indicated a significant model bias, wherein the classifier predominantly predicted a single, visually distinct class.

2.  **Improved Video-Level Model:** To address this limitation, the feature representation was enhanced. A video-level histogram was generated for each action sequence, summarizing the distribution of visual words across all its constituent frames.

Consequently, this improved methodology resulted in a substantial performance increase. The final model achieved a classification accuracy of **92.02%**. This demonstrates that a holistic feature representation capturing the aggregate properties of an entire video sequence is critical for robust and accurate action recognition.

---

### ## System Usage

#### 1. System Preparation

-   **Clone Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-GitHub-Username]/human-action-recognition.git
    cd human-action-recognition
    ```
-   **Dataset Acquisition:** Download the UCF-101 dataset and place the contents within a `data/raw/UCF-101` directory.
-   **Dependency Installation:**
    ```bash
    pip install -r requirements.txt
    ```

#### 2. Pipeline Execution

Execute the scripts from the `src/` directory in sequential order to replicate the full pipeline.

```bash
# Data Preparation
python src/01_filter_dataset.py
python src/02_extract_frames.py

# Vocabulary and Feature Generation
python src/04_build_vocabulary.py
python src/05b_create_video_histograms.py

# Model Training and Evaluation
python src/06b_train_on_video_features.py

# Inference on a New Video
python src/predict.py --video "path/to/video.avi"
```
