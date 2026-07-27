# utils_history.py
# ------------------------------------------------------------
# Keeps a running log of every image/video detection so users
# can review past results without re-uploading files.
#
# Storage strategy:
#   - st.session_state.detection_history -> fast, in-memory list
#     (newest first) used to render the table during the session.
#   - detection_history.csv on disk -> makes the log survive app
#     restarts / server reloads (Streamlit reruns don't wipe it).
# ------------------------------------------------------------

import os
import pandas as pd
import streamlit as st

HISTORY_FILE = "detection_history.csv"

HISTORY_COLUMNS = [
    "Timestamp",
    "Type",              # "Image" or "Video"
    "FileName",
    "Prediction",        # "Accident" / "NonAccident"
    "Confidence (%)",
    "Severity",
    "Processing Time (s)",
]


def init_history():
    """Load history into session_state once per session."""
    if "detection_history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            try:
                df = pd.read_csv(HISTORY_FILE)
                st.session_state.detection_history = df.to_dict("records")
            except Exception:
                st.session_state.detection_history = []
        else:
            st.session_state.detection_history = []


def log_detection(record: dict):
    """
    Add one detection result to the history (newest on top) and
    persist the full log to CSV.

    `record` should contain keys matching HISTORY_COLUMNS, e.g.:
        {
            "Timestamp": "27-07-2026 10:15:00",
            "Type": "Image",
            "FileName": "cctv_01.jpg",
            "Prediction": "Accident",
            "Confidence (%)": 92.13,
            "Severity": "Severity2 (Moderate)",
            "Processing Time (s)": 0.84,
        }
    """
    init_history()
    # Make sure every expected column is present
    clean_record = {col: record.get(col, "") for col in HISTORY_COLUMNS}
    st.session_state.detection_history.insert(0, clean_record)

    df = pd.DataFrame(st.session_state.detection_history, columns=HISTORY_COLUMNS)
    try:
        df.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        st.warning(f"Could not save detection history to disk: {e}")


def get_history_df() -> pd.DataFrame:
    """Return the current history as a DataFrame (newest first)."""
    init_history()
    if not st.session_state.detection_history:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(st.session_state.detection_history, columns=HISTORY_COLUMNS)


def clear_history():
    """Wipe both the in-memory and on-disk history."""
    st.session_state.detection_history = []
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except Exception as e:
            st.warning(f"Could not remove {HISTORY_FILE}: {e}")
