# 🏸 Badminton Shot Recognition using CNN-LSTM

Enhancing badminton performance analytics by automatically classifying shot types from video using pose estimation and a CNN-LSTM deep learning model.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Manual analysis of badminton footage is slow, subjective, and hard to scale. This project automates shot recognition by extracting body pose from video and classifying the movement sequence with a deep learning model.

The project is based on a research paper on badminton shot recognition that reported 93% accuracy. Our main contribution over the base paper is adding a **sixth shot class (Lob)**.

## 🎯 Shot Classes

| Label | Shot |
|-------|------|
| 0 | Smash |
| 1 | Clear |
| 2 | Drop |
| 3 | Drive |
| 4 | Net Shot |
| 5 | Lob |

## ⚙️ How It Works

```
Video → Frame Extraction → Pose Estimation (12 keypoints)
      → Feature Engineering (78 features) → CNN-LSTM → Shot Prediction
```

1. **Frame extraction:** OpenCV reads the video and samples frames.
2. **Pose estimation:** MediaPipe detects 12 body keypoints per frame.
3. **Feature engineering:** 78 features are computed from the keypoints (normalized coordinates, velocities, accelerations, and joint angles).
4. **Sequence classification:** A CNN-LSTM model learns spatial and temporal patterns across frame sequences.
5. **Output:** Predicted shot type with confidence scores, plus an annotated video.

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Deep learning:** TensorFlow, Keras
- **Computer vision:** OpenCV, MediaPipe, Pillow
- **Data processing:** NumPy, scikit-learn, tqdm
- **Visualization:** Matplotlib, Seaborn

## 📁 Project Structure

```
badminton_shot_recognition/
├── configs/                 # Configuration (config.yaml)
├── data/
│   ├── raw_videos/          # Input videos
│   ├── frames/              # Extracted frames
│   ├── keypoints/           # Detected keypoints (JSON)
│   └── processed/           # Feature datasets
├── models/                  # Saved trained models
├── outputs/                 # Predictions and visualizations
├── src/
│   ├── preprocessing/       # Video-to-frames, pose estimation
│   ├── feature_extraction/  # Feature engineering
│   ├── models/              # Training and inference
│   └── utils/               # Helper functions
├── notebooks/               # Jupyter notebooks
├── main.py                  # Pipeline entry point
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Usage

### Prepare data
1. Put your videos in `data/raw_videos/`.
2. Create `data/labels.json` mapping each video name to a shot label (0-5).

```json
{
  "video_smash_1": 0,
  "video_clear_1": 1
}
```

### Run the pipeline
```bash
# Full pipeline (frames → keypoints → features → training)
python main.py --step all --labels data/labels.json

# Or run step by step
python main.py --step frames
python main.py --step keypoints
python main.py --step features --labels data/labels.json
python main.py --step train
```

### Run inference on a new video
```bash
python main.py --step inference --video path/to/video.mp4
```

Outputs are saved to `outputs/predictions/` (annotated video and JSON predictions).

## 📊 Results

| Metric | Value |
|--------|-------|
| Test accuracy | _XX%_ |
| Precision (macro) | _XX%_ |
| Recall (macro) | _XX%_ |
| F1-score (macro) | _XX%_ |

_Add your confusion matrix and training curves here:_

```
![Confusion Matrix](outputs/visualizations/confusion_matrix.png)
![Training History](outputs/visualizations/training_history.png)
```

## 🔭 Scope

**In scope:** single-player shot classification for 6 shot types, end-to-end pipeline from video to prediction.

**Out of scope:** multi-player tracking, shuttlecock tracking, rally outcome prediction, 3D pose reconstruction.

## 🔮 Future Work

- Larger and more diverse dataset (more players, camera angles)
- Shuttlecock trajectory tracking
- Real-time inference
- Additional shot types and doubles play

## 👥 Team

- Dimpul
- Ajay Kumar
- Byreddy Rithwik
- Chaitanya
- Shiva Sai Rathod

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- Base research paper on badminton shot recognition
- [MediaPipe](https://developers.google.com/mediapipe) for pose estimation
- TensorFlow and Keras for deep learning
