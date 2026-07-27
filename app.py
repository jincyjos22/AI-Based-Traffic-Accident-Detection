# Step 1: Import Libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
import time
import base64
from datetime import datetime

from utils_history import init_history, log_detection, get_history_df, clear_history
from utils_pdf import generate_image_report_pdf, generate_video_report_pdf

# Step 2: Configure the Streamlit Page
st.set_page_config(
    page_title="AI Traffic Accident Monitoring System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Step 2b: Require login before anything else renders
from auth import require_login, current_user, logout
require_login()

# Step 2c: Initialize the persistent detection history log
init_history()

# Step 3: Background Image Function (safe version - checks file exists, called once)
def add_bg():
    bg_path = "images/background.png"
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.warning("Background image not found")


# Call background + glassmorphism styling once, at top level (not inside the function)
add_bg()

st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"]{
    background: rgba(255,255,255,0.05);
}

/* Main container */
.block-container{
    background: transparent !important;
    padding:2rem;
    border:none;
    border-radius:0;
    box-shadow:none;
    backdrop-filter:none;
}

/* Text */
h1,h2,h3,h4,h5,h6{
    color:white !important;
    text-shadow:2px 2px 8px black;
}

p,label,span{
    color:white !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:rgba(30,30,30,.9);
}

[data-testid="stSidebar"] *{
    color:white !important;
}
/* Logout Button */
div.stButton > button{
    width:100%;
    background:linear-gradient(90deg,#18c3ff,#0aa7e8) !important;
    color:white !important;
    border:none !important;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    padding:12px;
    outline:none !important;
    box-shadow:none !important;
    transition:0.3s;
}

/* Hover */
div.stButton > button:hover{
    background:linear-gradient(90deg,#33d1ff,#13b5f5) !important;
    color:white !important;
}

/* Click */
div.stButton > button:focus,
div.stButton > button:active{
    outline:none !important;
    box-shadow:none !important;
}
/* After Click */
div.stButton > button:focus,
div.stButton > button:focus-visible{
    background:linear-gradient(90deg,#18c3ff,#0aa7e8) !important;
    color:white !important;
    outline:none !important;
    box-shadow:none !important;
}
/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"]{
    background: transparent !important;
    border: none !important;
}

[data-testid="stFileUploaderDropzone"]{
    background: rgba(255,255,255,0.95) !important;
    border: 2px dashed #00C8FF !important;
    border-radius: 15px !important;
    padding: 30px !important;
}

[data-testid="stFileUploaderDropzone"] *{
    color:#111 !important;
}

[data-testid="stFileUploaderDropzone"] button{
    background:#00C8FF !important;
    color:white !important;
    border:none !important;
    border-radius:8px;
}

[data-testid="stFileUploaderDropzone"] svg{
    fill:#00C8FF !important;
}
/* Download Button */
.stDownloadButton > button{
    background: rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: white !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stDownloadButton > button:hover{
    background: rgba(255,255,255,0.18) !important;
    color: white !important;
}

.stDownloadButton > button:active,
.stDownloadButton > button:focus{
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    box-shadow: none !important;
}
/* Metrics */
[data-testid="stMetricValue"]{
    color:#007ACC !important;
}

[data-testid="stMetricLabel"]{
    color:#444 !important;
}

</style>
""", unsafe_allow_html=True)

# Step 5: Dashboard Title
st.markdown("""
<h1 style="text-align:center;">
🚦 AI Traffic Accident Monitoring System
</h1>

<h3 style="text-align:center;">
CCTV Accident Detection | Severity Prediction | AI Alert
</h3>
""", unsafe_allow_html=True)
# Project Banner
st.image(
    "images/project_banner.png",
    use_container_width=True
)
# Step 6: Sidebar
with st.sidebar:

    if os.path.exists("images/logo.png"):
        st.image("images/logo.png", width=120)

    st.title("Navigation")

    user = current_user()
    st.markdown(f"👤 **{user['display_name']}**")
    st.caption(f"Role: {user['role']}")

    st.markdown("---")

    if st.button("🔒 Logout", key="logout_btn", use_container_width=True):
        logout()

    st.markdown("---")

    st.success("🟢 System Online")
    st.write("✔ Image Detection")
    st.write("✔ Video Detection")
    st.write("✔ Severity Prediction")
    st.write("✔ Emergency Alert")
# Step 7: File Validation for Models
ACCIDENT_MODEL_PATH = "accident_detection_model.keras"
SEVERITY_MODEL_PATH = "severity_prediction_model.keras"

missing = [p for p in (ACCIDENT_MODEL_PATH, SEVERITY_MODEL_PATH) if not os.path.exists(p)]
if missing:
    st.error(
        "❌ Model file(s) not found: " + ", ".join(missing) +
        ". Please place both .keras files in the same folder as app.py."
    )
    st.stop()

# Step 8: Load Models (Cached)
@st.cache_resource
def load_models():
    accident = load_model(ACCIDENT_MODEL_PATH)
    severity = load_model(SEVERITY_MODEL_PATH)
    return accident, severity

accident_model, severity_model = load_models()

IMG_SIZE = (224, 224)

# These MUST match the alphabetical folder order Keras used when it built
# train_ds.class_names / severity_train.class_names in your training notebook.
# Print those in Colab and confirm before trusting these lists.
ACCIDENT_CLASSES = ["Accident", "NonAccident"]      # index 0, 1
SEVERITY_CLASSES = {
    0: "Severity1 (Minor)",
    1: "Severity2 (Moderate)",
    2: "Severity3 (Severe)"
}  # index 0, 1, 2


# ---- Shared preprocessing function used by BOTH image and video paths ----
# Matches training: tf.keras.layers.Rescaling(1./255) was used in the notebook,
# NOT mobilenet_v2.preprocess_input - so we rescale to [0,1] here to match.
def preprocess_array(img_array):
    img_array = img_array.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_accident(img_array):

    preds = accident_model.predict(img_array, verbose=0)[0]

    print("Raw predictions:", preds)
    print("Argmax:", np.argmax(preds))

    class_idx = np.argmax(preds)
    confidence = float(preds[class_idx])
    label = ACCIDENT_CLASSES[class_idx]

    print("Label:", label)

    return label, confidence
def predict_severity(img_array):
    """Runs the severity model. Returns (label, confidence)."""
    preds = severity_model.predict(img_array, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = SEVERITY_CLASSES[class_idx]
    return label, confidence


def severity_display(label):
    """Adds a color-coded icon to the raw severity class name."""
    icons = {
        "Severity1": "🟢 Severity1 (Minor)",
        "Severity2": "🟡 Severity2 (Moderate)",
        "Severity3": "🔴 Severity3 (Severe)",
    }
    return icons.get(label, label)


# ============================================================
# IMAGE DETECTION SECTION
# ============================================================

st.markdown("---")

st.markdown("""
<div style="
background:rgba(0,0,0,0.55);
padding:15px;
border-radius:12px;
margin-bottom:15px;">
<h2 style="
color:#00E5FF;
margin:0;
font-size:32px;
font-weight:bold;
text-shadow:2px 2px 6px black;">
📷 CCTV Image Accident Detection
</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h4 style='color:white;'>Upload CCTV Image</h4>",
    unsafe_allow_html=True
)

# Initialize uploader key
if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0

uploaded_image = st.file_uploader(
    "📤 Drag and drop a CCTV image or click Browse Files",
    type=["jpg", "jpeg", "png"],
    key=f"image_upload_{st.session_state.image_uploader_key}"
)

if st.button("🔄 Clear Image", use_container_width=True):
    st.session_state.image_uploader_key += 1
    st.rerun()

if uploaded_image:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(image, caption="Uploaded CCTV Image", width=500)

    # Preprocess
    processed_image = image.resize(IMG_SIZE)
    image_array = np.array(processed_image)
    image_array = preprocess_array(image_array)

    # Predict - Accident Detection Model
    start_time = time.time()

    with st.spinner("🔍 Analyzing CCTV Image..."):
        accident_label, confidence = predict_accident(image_array)

    end_time = time.time()

    status = accident_label

    if status == "Accident":
        result = " Accident Detected"
    else:
        result = " No Accident"

    st.markdown(
f"""
<div style="
background:rgba(0,0,0,0.75);
padding:15px;
border-radius:15px;
color:white;
font-size:28px;
font-weight:bold;
border:2px solid #00C8FF;
box-shadow:0px 4px 15px rgba(0,0,0,0.3);
">

📊 Detection Result

</div>
""",
unsafe_allow_html=True
    )

    if status == "Accident":
        st.error(f"""
🚨 **Accident Detected**

**Confidence:** {confidence*100:.2f}%

**Processing Time:** {end_time-start_time:.2f} seconds
""")
    else:
        st.success(f"""
✅ **No Accident Detected**

**Confidence:** {confidence*100:.2f}%

**Processing Time:** {end_time-start_time:.2f} seconds
""")

    # Confidence Progress Bar
    st.subheader("🎯 Confidence Score")
    confidence_percentage = int(confidence * 100)
    st.progress(confidence_percentage)
    st.write(f"Model Confidence: {confidence_percentage}%")

    # Severity - only run the severity model if an accident was detected
    if status == "Accident":
        severity_label, severity_confidence = predict_severity(image_array)
        severity = severity_display(severity_label)
    else:
        severity_label, severity_confidence = None, None
        severity = "No Accident"

    st.subheader("⚠️ Severity Assessment")
    if status == "Accident":
        st.info(f"Estimated Severity: {severity} (confidence: {severity_confidence*100:.2f}%)")
    else:
        st.info(f"Estimated Severity: {severity}")

    # Emergency Alert Banner
    if status == "Accident":
        st.markdown(
            """
            <div style="
            background-color:red;
            padding:25px;
            border-radius:15px;
            text-align:center;
            color:white;
            font-size:30px;
            font-weight:bold;
            ">
            🚨 EMERGENCY ALERT 🚨<br>
            ACCIDENT DETECTED
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.success("✅ Road condition appears normal")

    # Dashboard Metrics
    st.subheader("📡 AI Monitoring Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Prediction", value=status)
    with col2:
        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
    with col3:
        st.metric(label="Severity", value=severity)
    with col4:
        st.metric(label="Inference Time", value=f"{end_time-start_time:.2f} sec")

    st.caption(f"🕒 Detection Time: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    # Two Column View
    col_img, col_result = st.columns(2)
    with col_img:
        st.image(image, caption="📷 CCTV Input Image", use_container_width=True)
    with col_result:
        st.write("### AI Prediction")
        st.write(f"**Result:** {result}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.write(f"**Severity:** {severity}")

    # ============================================================
    # IMAGE DETECTION REPORT DOWNLOAD
    # ============================================================

    report_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    image_report = f"""
AI TRAFFIC ACCIDENT MONITORING SYSTEM
=====================================

ACCIDENT DETECTION REPORT

Date & Time:
{report_time}

Prediction:
{result}

Confidence:
{confidence*100:.2f}%

Severity:
{severity}

Processing Time:
{end_time-start_time:.2f} seconds

Emergency Alert:
{"ACTIVATED" if status=="Accident" else "NOT REQUIRED"}
"""

    
    # ---- Save a temp copy of the image so the PDF can embed it ----
    temp_image_path = "temp_report_image.jpg"
    image.save(temp_image_path)

    pdf_bytes = generate_image_report_pdf(
        image_path=temp_image_path,
        result=result,
        confidence=confidence,
        severity=severity,
        processing_time=end_time - start_time,
        status=status,
    )

    st.download_button(
        label="🧾 Download Image Detection Report (PDF)",
        data=pdf_bytes,
        file_name="image_accident_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    # ---- Log this detection to history ----
    log_detection({
        "Timestamp": report_time,
        "Type": "Image",
        "FileName": uploaded_image.name,
        "Prediction": status,
        "Confidence (%)": round(confidence * 100, 2),
        "Severity": severity,
        "Processing Time (s)": round(end_time - start_time, 2),
    })

    
# ============================================================
# VIDEO DETECTION SECTION
# ============================================================
st.markdown("---")

st.markdown("""
<div style="
background:rgba(0,0,0,0.55);
padding:15px;
border-radius:12px;
margin-bottom:15px;">
<h2 style="
color:#00E5FF;
margin:0;
font-size:32px;
font-weight:bold;
text-shadow:2px 2px 6px black;">
🎥 CCTV Video Accident Detection
</h2>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h4 style='color:white;'>Upload CCTV Video</h4>",
    unsafe_allow_html=True
)

if "video_uploader_key" not in st.session_state:
    st.session_state.video_uploader_key = 0

uploaded_video = st.file_uploader(
    "",
    type=["mp4", "avi", "mov"],
    key=f"video_upload_{st.session_state.video_uploader_key}"
)

if st.button("🔄 Clear Video", use_container_width=True):
    st.session_state.video_uploader_key += 1
    st.rerun()

# Initialize variables
accident_detected = False
output_path = None
video_playable = False
video_path = None
worst_severity_label = None
max_confidence = 0.0
first_accident_time = None

FRAME_SKIP = 10

if uploaded_video:

    video_path = "temp_video.mp4"
    output_path = "processed_video.mp4"

    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Could not open the uploaded video.")

    else:

        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            total_frames = 1

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        accident_detected = False
        max_confidence = 0.0
        first_accident_time = None
        worst_severity_label = None

        last_label = "NonAccident"
        last_confidence = 0.0
        last_severity_label = None

        frame_count = 0
        severity_rank = {
        "Severity1 (Minor)": 1,
        "Severity2 (Moderate)": 2,
         "Severity3 (Severe)": 3

        }

        st.info("🔍 Analyzing CCTV Video...")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count % FRAME_SKIP == 0:

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, IMG_SIZE)

                frame_array = preprocess_array(np.array(resized))

                label, confidence = predict_accident(frame_array)

                last_label = label
                last_confidence = confidence

                if label == "Accident":

                    accident_detected = True

                    if confidence > max_confidence:
                        max_confidence = confidence

                    if first_accident_time is None:
                        first_accident_time = round(frame_count / fps, 2)

                    sev_label, sev_conf = predict_severity(frame_array)

                    last_severity_label = sev_label

                    if (worst_severity_label is None or
                            severity_rank[sev_label] >
                            severity_rank[worst_severity_label]):

                        worst_severity_label = sev_label

                else:
                    last_severity_label = None

            # Decide what to display
            if accident_detected:
                display_label = "Accident"
                display_confidence = max_confidence
                display_severity = worst_severity_label
            else:
                display_label = last_label
                display_confidence = last_confidence
                display_severity = last_severity_label

            text = f"{display_label} ({display_confidence*100:.1f}%)"

            if display_severity:
                text += f" | {display_severity}"

            color = (0, 0, 255) if display_label == "Accident" else (0, 255, 0)

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            frame_placeholder.image(frame, channels="BGR")

            out.write(frame)

            progress_bar.progress(min(frame_count / total_frames, 1.0))

            status_text.write(
                f"Processed {frame_count}/{total_frames} Frames"
            )

        cap.release()
        out.release()

        print("Output exists:", os.path.exists(output_path))

        if os.path.exists(output_path):
            print("Output size:", os.path.getsize(output_path), "bytes")

        # ---- Re-encode to H.264 so browsers can actually play it ----
        # cv2.VideoWriter's "mp4v" fourcc produces a codec most browsers
        # (Chrome, Edge, etc.) refuse to play in an HTML5 <video> tag,
        # which is why st.video() shows "No video with supported format
        # and MIME type found." ffmpeg re-encodes it to H.264/yuv420p,
        # which every browser supports.
        h264_path = "processed_video_h264.mp4"
        try:
            import subprocess
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", output_path,
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    h264_path,
                ],
                check=True,
                capture_output=True,
            )
            output_path = h264_path
            video_playable = True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print("ffmpeg re-encode failed:", e)

        st.markdown(
             "<h3 style='color:white;'>📊 Video Detection Result</h3>",
              unsafe_allow_html=True
        )

        if accident_detected:

            severity = severity_display(worst_severity_label)

            st.error(f"""
🚨 Accident Detected

Peak Confidence: {max_confidence*100:.2f}%

First Detected: {first_accident_time} sec

Severity: {severity}
""")

        else:
            st.success("✅ No Accident Detected")

        # ---- Log this detection to history ----
        video_severity = severity_display(worst_severity_label) if worst_severity_label else "No Accident"
        log_detection({
            "Timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "Type": "Video",
            "FileName": uploaded_video.name,
            "Prediction": "Accident" if accident_detected else "NonAccident",
            "Confidence (%)": round(max_confidence * 100, 2) if accident_detected else 0.0,
            "Severity": video_severity,
            "Processing Time (s)": "-",
        })

        # ---- PDF report download ----
        video_pdf_bytes = generate_video_report_pdf(
            video_name=uploaded_video.name,
            accident_detected=accident_detected,
            max_confidence=max_confidence,
            first_accident_time=first_accident_time,
            severity=video_severity,
        )

        st.download_button(
            label="🧾 Download Video Detection Report (PDF)",
            data=video_pdf_bytes,
            file_name="video_accident_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ============================================================
# SHOW & DOWNLOAD PROCESSED VIDEO
# ============================================================

if uploaded_video and output_path and os.path.exists(output_path):

    st.subheader("🎥 Processed CCTV Video")
    # Read the processed video as bytes
    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()

    # Display processed video (only if we know it's in a browser-playable
    # format — otherwise skip straight to the download button)
    if video_playable:
        st.video(video_bytes)
    else:
        st.info(
            "🎬 Preview isn't available in this browser, but your "
            "processed video is ready — use the download button below "
            "to save and watch it.\n\n"
            "(Tip: install ffmpeg on this machine to enable in-browser "
            "preview next time.)"
        )

    # Download button
    st.download_button(
        label="⬇ Download Processed Video",
        data=video_bytes,
        file_name="processed_video.mp4",
        mime="video/mp4",
        use_container_width=True
    )

else:
    st.error("❌ Processed video could not be displayed.")
# Delete only the uploaded temporary video
if uploaded_video and video_path and os.path.exists(video_path):
    os.remove(video_path)

# Keep the processed video so it can be viewed and downloaded
# (Do not delete output_path here.)

# ============================================================
# DETECTION HISTORY SECTION
# ============================================================

st.markdown("---")

st.markdown("""
<div style="
background:rgba(0,0,0,0.55);
padding:15px;
border-radius:12px;
margin-bottom:15px;">
<h2 style="
color:#00E5FF;
margin:0;
font-size:32px;
font-weight:bold;
text-shadow:2px 2px 6px black;">
📜 Detection History
</h2>
</div>
""", unsafe_allow_html=True)

history_df = get_history_df()

if history_df.empty:
    st.info("No detections logged yet. Run an image or video detection above to start building history.")
else:
    type_filter = st.selectbox(
        "Filter by type",
        options=["All", "Image", "Video"],
        key="history_type_filter"
    )

    display_df = history_df if type_filter == "All" else history_df[history_df["Type"] == type_filter]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.download_button(
            label="⬇ Download Full History (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="detection_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_h2:
        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history()
            st.rerun()

# ============================================================
# SMART CITY MAP SECTION
# ============================================================

st.markdown("---")

st.markdown("""
<div style="
background:rgba(0,0,0,0.55);
padding:20px;
border-radius:15px;
margin-bottom:20px;
">

<h2 style="
color:#00E5FF;
margin-top:0;
text-shadow:2px 2px 6px black;">
🗺️ Smart City Traffic Monitoring
</h2>

<h3 style="color:white;">
📍 Accident Location Tracking
</h3>

<p style="color:white;font-size:20px;">
<b>Future Integration:</b>
</p>

<ul style="
color:white;
font-size:18px;
line-height:2;">
<li>📍 GPS Location from CCTV Camera</li>
<li>🗺️ Google Maps API Integration</li>
<li>🚑 Emergency Service Routing</li>
<li>🚦 Traffic Control Center Notification</li>
</ul>

</div>
""", unsafe_allow_html=True)

# Map Image
if os.path.exists("images/map_placeholder.png"):
    st.image(
        "images/map_placeholder.png",
        caption="Future Smart City Accident Location Map",
        use_container_width=True
    )
else:
    st.warning("Map placeholder image not found")

# Status Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📷 Camera ID", "CCTV_001")

with col2:
    st.metric("📍 Location", "Main Road")

with col3:
    st.metric("🚦 Status", "Monitoring")
# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
"""
<div style="
text-align:center;
color:white;
font-size:16px;
">
<h3>🚦 AI Traffic Accident Monitoring System</h3>
<p>Deep Learning Based CCTV Accident Detection</p>
<p>Developed by <b>Jincy M.S</b></p>
<p>Using MobileNet Transfer Learning | Streamlit Dashboard</p>
<p>© 2026 All Rights Reserved</p>
</div>
""",
unsafe_allow_html=True
)

st.success("🟢 AI Monitoring System Running Successfully")
