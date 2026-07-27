# 🚦 AI-Based Traffic Accident Detection

An AI-powered CCTV monitoring dashboard that detects road accidents from images and video, predicts accident severity, and raises emergency alerts in real time. Built with **MobileNetV2 transfer learning**, **TensorFlow/Keras**, and a **Streamlit** dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

---

## 📖 Overview

This project uses deep learning to monitor traffic CCTV feeds and automatically:

1. **Detect accidents** — classifies a frame/image as `Accident` or `NonAccident`.
2. **Predict severity** — if an accident is detected, a second model classifies it as `Severity 1 (Minor)`, `Severity 2 (Moderate)`, or `Severity 3 (Severe)`.
3. **Raise alerts** — displays an on-screen emergency alert banner when an accident is detected.
4. **Log and report** — every detection is logged to a history table and can be exported as a PDF report or CSV.

The models are trained using **MobileNetV2 transfer learning** on a labeled accident image dataset, with data augmentation and class-weight balancing to handle class imbalance. The training pipeline also includes **Grad-CAM** visualizations for explainability, highlighting the image regions that most influenced each prediction.

---

## ✨ Features

- 🔐 **Login-protected dashboard** (session-based authentication)
- 📷 **Image detection** — upload a CCTV image and get an instant accident/no-accident prediction with confidence score
- 🎥 **Video detection** — upload CCTV footage; frames are sampled and analyzed, with live annotated video preview and a downloadable processed video
- ⚠️ **Severity prediction** — automatically triggered whenever an accident is detected
- 🚨 **Emergency alert banner** for detected accidents
- 🧾 **PDF report generation** for both image and video detections
- 📜 **Detection history** — filterable table of past detections with CSV export and clear/reset option
- 🗺️ **Smart City Map section** — placeholder UI for future GPS/Google Maps integration and emergency service routing
- 🎨 Custom glassmorphism-styled Streamlit UI with background image support

---

## 🧠 Model Architecture

Both models use **MobileNetV2** (ImageNet weights) as a feature extractor, with a custom classification head:

| Model | Input Size | Output Classes | Notes |
|---|---|---|---|
| `accident_detection_model.keras` | 224×224×3 | `Accident`, `NonAccident` | Fine-tuned (last 30 layers unfrozen) |
| `severity_prediction_model.keras` | 224×224×3 | `Severity1`, `Severity2`, `Severity3` | Base model frozen, trained on accident-only subset |

**Training pipeline highlights** (see the training notebook):
- Data augmentation: random flip, rotation, zoom, contrast, translation
- Pixel rescaling to `[0, 1]` (`Rescaling(1./255)`)
- Class-weight balancing via `sklearn.utils.class_weight.compute_class_weight`
- `EarlyStopping` + `ModelCheckpoint` callbacks, best-model restoration
- Evaluation via confusion matrix, classification report, ROC/precision-recall curves
- **Grad-CAM** heatmaps for explainable predictions

---

## 📁 Project Structure

```
AI-Based-Traffic-Accident-Detection/
├── app.py                             # Streamlit dashboard (main entry point)
├── auth.py                            # Login / session authentication (required, not included above)
├── utils_history.py                   # Detection history logging & CSV export helpers
├── utils_pdf.py                       # PDF report generation helpers
├── accident_detection_model.keras     # Trained accident detection model
├── severity_prediction_model.keras    # Trained severity prediction model
├── images/
│   ├── background.png
│   ├── project_banner.png
│   ├── logo.png
│   └── map_placeholder.png
├── Traffic_accident_detection.ipynb   # Model training notebook (Colab)
└── README.md
```

> **Note:** `auth.py`, `utils_history.py`, and `utils_pdf.py` are imported by `app.py` and must be present in the project root. The two `.keras` model files must also sit alongside `app.py` — the app checks for them at startup and will show an error if either is missing.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/jincyjos22/AI-Based-Traffic-Accident-Detection.git
cd AI-Based-Traffic-Accident-Detection
pip install -r requirements.txt
```

**Core dependencies:**

```
streamlit
tensorflow
opencv-python
pillow
numpy
pandas
```

### Add the trained models

Download or train `accident_detection_model.keras` and `severity_prediction_model.keras` (see [Training](#-training-the-models) below) and place both files in the project root, next to `app.py`.

### Run the app

```bash
streamlit run app.py
```

Streamlit will print URLs you can open in your browser, for example:

```
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.8:8501
```

- Use the **Local URL** to access the dashboard from the same machine.
- Use the **Network URL** to access it from another device on the same network (e.g. a phone or another PC).

---

## 🖥️ Usage

1. **Log in** through the authentication screen.
2. **Image detection:** upload a CCTV image (`.jpg`, `.jpeg`, `.png`) to get an accident prediction, confidence score, severity assessment (if applicable), and a downloadable PDF report.
3. **Video detection:** upload CCTV footage (`.mp4`, `.avi`, `.mov`). The app samples frames, overlays live predictions, and produces an annotated, downloadable processed video plus a PDF summary.
4. **Detection history:** review, filter, and export all past detections, or clear the log.

---

## 🎓 Training the Models

The full training pipeline is provided in `Traffic_accident_detection.ipynb` (originally built in Google Colab). At a high level it:

1. Mounts a Google Drive dataset and extracts accident/non-accident image folders.
2. Builds a binary image dataset (`Accident` vs `NonAccident`) with an 80/20 train/validation split.
3. Applies augmentation and normalization, then fine-tunes a MobileNetV2-based classifier.
4. Evaluates with confusion matrices, classification reports, and Grad-CAM visualizations.
5. Repeats a similar pipeline on an accident-only, severity-labeled dataset to train the severity model.
6. Saves both models as `accident_detection_model.keras` and `severity_prediction_model.keras`.

To retrain, update the dataset paths to your own data and run the notebook cells in order.

> ⚠️ **Important:** the class index order the deployed app uses (`ACCIDENT_CLASSES` and `SEVERITY_CLASSES` in `app.py`) must exactly match the alphabetical folder order Keras used when building `train_ds.class_names` / `severity_train.class_names` during training. Confirm these in the notebook output before deploying a retrained model.

---

## 🔮 Roadmap

- [ ] Live GPS location tagging from CCTV cameras
- [ ] Google Maps API integration for accident location visualization
- [ ] Automated emergency service routing
- [ ] Traffic control center notification integration
- [ ] Multi-camera / multi-feed monitoring

---

## 👩‍💻 Author

**Jincy M.S**

## 📄 License

© 2026 All Rights Reserved.
