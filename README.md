# 🚦 AI-Based Traffic Accident Detection

An AI-powered traffic accident detection system that analyzes CCTV images and videos to detect road accidents, classify accident severity, and generate detection reports. The application is developed using MobileNetV2 Transfer Learning, TensorFlow/Keras, and Streamlit, providing an interactive web interface for accident analysis from uploaded CCTV images and videos.
Technologies:
• Python • TensorFlow • Keras • MobileNetV2 • Streamlit • OpenCV • NumPy • Pandas

---

# 🌐 Live Demo

**Application URL:**

https://ai-based-traffic-accident-detection-pdqrolpn4chxsrw9pnt62h.streamlit.app/

# 💻 GitHub Repository

https://github.com/jincyjos22/AI-Based-Traffic-Accident-Detection

# 🔐 Demo Login

For testing and academic project evaluation:

**Username:** `admin` 
**Password:** `Admin@123`

> This demo account is provided for academic project evaluation purposes only.

---

# 📖 Overview

Road traffic accidents require immediate detection and response to reduce casualties and improve emergency management.

This project uses deep learning to automatically analyze traffic CCTV images and videos and:

* Detect whether an accident has occurred.
* Classify the accident as Accident or Non-Accident.
* Predict accident severity into:

  * Severity 1 (Minor)
  * Severity 2 (Moderate)
  * Severity 3 (Severe)
* Display emergency alerts.
* Generate downloadable PDF reports.
* Maintain a history of predictions.

The application is implemented as an interactive Streamlit dashboard.

---

# ✨ Features

* Login-protected dashboard
* Image accident detection
* Video accident detection
* MobileNetV2 Transfer Learning
* Automatic severity prediction
* Emergency alert notification
* Confidence score display
* PDF report generation
* Detection history
* CSV export
* Streamlit-based responsive interface
* Smart City Map placeholder for future integration

---

# 🧠 Model Architecture

## Accident Detection Model

* Base Model: MobileNetV2
* Input Size: 224 × 224 × 3
* Output Classes:

  * Accident
  * NonAccident

Transfer Learning with fine-tuning of the final layers is used for high prediction accuracy.

## Severity Prediction Model

A second MobileNetV2 model predicts accident severity only if an accident is detected.

Output Classes:

* Severity 1 (Minor)
* Severity 2 (Moderate)
* Severity 3 (Severe)

| Model | Input Size | Output Classes | Notes |
|---|---|---|---|
| `accident_detection_model.keras` | 224×224×3 | `Accident`, `NonAccident` | Fine-tuned (last 30 layers unfrozen) |
| `severity_prediction_model.keras` | 224×224×3 | `Severity1`, `Severity2`, `Severity3` | Base model frozen, trained on accident-only subset |

---

# ⚙️ Technologies Used

* Python
* Streamlit
* TensorFlow
* Keras
* MobileNetV2
* OpenCV
* NumPy
* Pandas
* Pillow
* Matplotlib
* Scikit-learn

---

# 📂 Dataset

This project uses two datasets:

### 1. Accident Detection Dataset

Contains Accident and Non-Accident images.

### 2. Accident Severity Dataset

Contains accident images categorized into:

* Severity 1
* Severity 2
* Severity 3

> Due to GitHub storage limitations, the datasets are not included in this repository.

**Dataset Download**

https://www.kaggle.com/datasets/suryaprabhakaran2005/road-accidents-from-cctv-footages-dataset?utm_source=chatgp

---

# 📁 Project Structure

```text
AI-Based-Traffic-Accident-Detection/

│── app.py
│── auth.py
│── utils_history.py
│── utils_pdf.py
│── requirements.txt
│── README.md
│── .gitignore

│── accident_detection_model.keras
│── severity_prediction_model.keras

│── images/
│     ├── background.png
│     ├── logo.png
│     ├── project_banner.png
│     └── map_placeholder.png

│── Traffic_accident_detection.ipynb
```


---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/jincyjos22/AI-Based-Traffic-Accident-Detection.git
```

Go to the project folder

```bash
cd AI-Based-Traffic-Accident-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application opens in your browser. Streamlit will print URLs you can use, for example:

```
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.8:8501
```

* Use the **Local URL** to access the dashboard from the same machine.
* Use the **Network URL** to access it from another device on the same network (e.g. a phone or another PC).

---

# 🖥️ Usage

1. **Log in** through the authentication screen.
2. **Image Detection:** Upload a clear road traffic or CCTV image (`.jpg`, `.jpeg`, `.png`). The application analyzes the image to:
   - Detect whether an accident has occurred.
   - Display the prediction and confidence score.
   - Predict accident severity (only if an accident is detected).
   - Generate a downloadable PDF report.

   **Recommended images:**
   - ✅ CCTV or road traffic images
   - ✅ Clear accident or non-accident road scenes
   - ✅ Front, side, or rear vehicle collisions
   - ✅ Minor, moderate, or severe accidents

   **Avoid:**
   - ❌ Blurry or low-resolution images
   - ❌ Cartoon or AI-generated images
   - ❌ Images showing only a small portion of a damaged vehicle
   - ❌ Images without vehicles or road scenes
3. **Video detection:** upload CCTV footage (`.mp4`, `.avi`, `.mov`). The app samples frames, overlays live predictions, and produces an annotated, downloadable processed video plus a PDF summary.
4. **Detection history:** review, filter, and export all past detections, or clear the log.

---

# 📸 Screenshots

## Login Page

<img width="1142" height="557" alt="image" src="https://github.com/user-attachments/assets/15da63bb-fc5d-4c67-be88-422ff4b376ca" />


---

## Dashboard

<img width="1862" height="887" alt="image" src="https://github.com/user-attachments/assets/9642bd51-9494-4618-8564-a97610693923" />


---

## Accident Detection

<img width="1476" height="811" alt="image" src="https://github.com/user-attachments/assets/496f1f7e-062c-4b11-b1d3-b80d7098edc4" />

---

## Severity Prediction

<img width="1352" height="772" alt="image" src="https://github.com/user-attachments/assets/f2313c8a-8c7c-4ab5-9957-6f080034e3e8" />


---

## Video Detection

<img width="1512" height="652" alt="image" src="https://github.com/user-attachments/assets/af3c9f91-cdaa-4f00-8614-787b8ef39945" />


---

# 📊 Prediction Workflow

```
Upload Image / Video
          │
          ▼
 Accident Detection Model
          │
     Accident?
     │        │
    No       Yes
     │        │
Display     Severity Model
Result          │
                ▼
Severity 1 / Severity 2 / Severity 3
```

---

# 📈 Model Training

The models are trained using MobileNetV2 Transfer Learning.

Training includes:

* Image preprocessing
* Data augmentation
* Image normalization
* Class balancing
* Early stopping
* Model checkpointing
* Fine-tuning
* Performance evaluation
* Grad-CAM visualization

The training notebook is available as:

```
Traffic_accident_detection.ipynb
```

At a high level, the training pipeline:

1. Mounts a Google Drive dataset and extracts accident/non-accident image folders.
2. Builds a binary image dataset (`Accident` vs `NonAccident`) with an 80/20 train/validation split.
3. Applies augmentation (random flip, rotation, zoom, contrast, translation) and pixel normalization (`Rescaling(1./255)`).
4. Fine-tunes a MobileNetV2-based classifier (last 30 layers unfrozen) with class-weight balancing to handle class imbalance.
5. Evaluates with confusion matrices, classification reports, ROC/precision-recall curves, and **Grad-CAM** heatmaps for explainability.
6. Repeats a similar pipeline on an accident-only, severity-labeled dataset to train the severity model (base model frozen).
7. Saves both models as `accident_detection_model.keras` and `severity_prediction_model.keras`.

---

# 📄 Reports

The application automatically generates:

* PDF prediction report
* Detection history
* CSV export

---

# 🔮 Future Improvements

* Live CCTV streaming
* Real-time accident monitoring
* Google Maps integration
* GPS location tracking
* Emergency service notification
* Cloud deployment improvements
* Multi-camera support

---

# 📦 Requirements

Main libraries:

```
streamlit
tensorflow
numpy
opencv-python
pandas
pillow
matplotlib
scikit-learn
```

---

# 👩‍💻 Author

**Jincy M.S.**

Bachelor of Computer Applications (BCA)

AI-Based Traffic Accident Detection Project

---

# 📜 License

This project is developed for educational and academic purposes.

Copyright © 2026 Jincy M.S.

All Rights Reserved.

---

# ⭐ Acknowledgements

* TensorFlow
* Keras
* Streamlit
* OpenCV
* MobileNetV2
* Google Colab
* Python Community
