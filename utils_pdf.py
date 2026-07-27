# utils_pdf.py
# ------------------------------------------------------------
# Generates branded PDF accident-detection reports (image + video)
# using fpdf2. Returns raw bytes so they can be wired straight into
# st.download_button(..., mime="application/pdf").
#
# Install requirement:
#   pip install fpdf2
# ------------------------------------------------------------

import os
from datetime import datetime

from fpdf import FPDF

BRAND_BLUE = (0, 122, 204)
ACCENT_RED = (200, 0, 0)
ACCENT_GREEN = (0, 150, 0)
GREY = (100, 100, 100)


class DetectionPDF(FPDF):
    """Adds a shared header/footer to every report page."""

    def header(self):
        if os.path.exists("images/logo.png"):
            self.image("images/logo.png", x=10, y=8, w=18)

        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*BRAND_BLUE)
        self.cell(0, 10, "AI Traffic Accident Monitoring System", ln=True, align="C")

        self.set_font("Helvetica", "", 11)
        self.set_text_color(*GREY)
        self.cell(0, 8, "Accident Detection Report", ln=True, align="C")

        self.ln(2)
        self.set_draw_color(0, 200, 255)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        stamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        self.cell(0, 10, f"Generated {stamp}  |  Page {self.page_no()}", align="C")


def _pdf_bytes(pdf: FPDF) -> bytes:
    """Normalize fpdf2's output() across versions to plain bytes."""
    out = pdf.output()
    if isinstance(out, (bytearray, bytes)):
        return bytes(out)
    return out.encode("latin-1")  # older fpdf returns a str


def _field(pdf: FPDF, label: str, value, color=(0, 0, 0)):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(55, 8, f"{label}:")

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*color)

    # Convert to string
    value = str(value)

    # Remove unsupported Unicode characters (emojis)
    for emoji in ["✅", "❌", "🚨", "⚠️", "📄", "🚗", "🎥", "📷"]:
        value = value.replace(emoji, "")

    pdf.cell(0, 8, value.strip(), ln=True)

def generate_image_report_pdf(
    image_path: str,
    result: str,
    confidence: float,
    severity: str,
    processing_time: float,
    status: str,
) -> bytes:
    """
    Build a one-page PDF for an image detection.

    image_path      - path to a saved copy of the uploaded/analyzed image
                       (pass None/"" to skip embedding the picture)
    result          - e.g. "Accident Detected" / "No Accident"
    confidence      - 0-1 float
    severity        - display string, e.g. "Severity2 (Moderate)"
    processing_time - seconds, float
    status          - "Accident" or "NonAccident" (drives the accent color)
    """
    pdf = DetectionPDF()
    pdf.add_page()

    if image_path and os.path.exists(image_path):
        pdf.image(image_path, x=55, w=100)
        pdf.ln(6)

    accent = ACCENT_RED if status == "Accident" else ACCENT_GREEN

    _field(pdf, "Date & Time", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    _field(pdf, "Prediction", result, accent)
    _field(pdf, "Confidence", f"{confidence * 100:.2f}%")
    _field(pdf, "Severity", severity)
    _field(pdf, "Processing Time", f"{processing_time:.2f} seconds")
    _field(
        pdf,
        "Emergency Alert",
        "ACTIVATED" if status == "Accident" else "NOT REQUIRED",
        accent,
    )

    return _pdf_bytes(pdf)


def generate_video_report_pdf(
    video_name: str,
    accident_detected: bool,
    max_confidence: float,
    first_accident_time,
    severity: str,
) -> bytes:
    """Build a one-page PDF summarizing a video detection run."""
    pdf = DetectionPDF()
    pdf.add_page()

    accent = ACCENT_RED if accident_detected else ACCENT_GREEN

    _field(pdf, "Date & Time", datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    _field(pdf, "Video File", video_name or "N/A")
    _field(
        pdf,
        "Result",
        "Accident Detected" if accident_detected else "No Accident Detected",
        accent,
    )

    if accident_detected:
        _field(pdf, "Peak Confidence", f"{max_confidence * 100:.2f}%")
        _field(pdf, "First Detected At", f"{first_accident_time} sec")
        _field(pdf, "Severity", severity)

    _field(
        pdf,
        "Emergency Alert",
        "ACTIVATED" if accident_detected else "NOT REQUIRED",
        accent,
    )

    return _pdf_bytes(pdf)
