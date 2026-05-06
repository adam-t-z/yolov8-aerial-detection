import os
import time

import cv2
import numpy as np
import streamlit as st
import yaml
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
os.environ["YOLO_VERBOSE"] = "False"

st.set_page_config(page_title="YOLO Detector", layout="wide")

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model_path = config["deployment"]["model_path"]
    return YOLO(model_path)


model = load_model()


# -----------------------------
# DRAW FUNCTION (BIG LABELS)
# -----------------------------
def draw_boxes(image, result, show_conf=True):
    annotated = image.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        label = model.names[cls]
        if show_conf:
            label += f" {conf:.2f}"

        # BIG BOX
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # BIG TEXT STYLE
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # background box
        cv2.rectangle(
            annotated,
            (x1, y1 - h - 12),
            (x1 + w + 12, y1),
            (0, 255, 0),
            -1,
        )

        # text
        cv2.putText(
            annotated,
            label,
            (x1 + 6, y1 - 6),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return annotated


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Settings")

input_source = st.sidebar.radio("Input Source", ["Image", "Video"])

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
show_conf = st.sidebar.checkbox("Show Confidence", True)

st.title("🎯 YOLO Object Detection App")
st.markdown("Upload an image or video to detect objects.")


# -----------------------------
# IMAGE MODE
# -----------------------------
if input_source == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = model(image, conf=confidence)[0]
        output = draw_boxes(image, result, show_conf)

        st.image(output, width="stretch")
        st.success(f"Detections: {len(result.boxes)}")


# -----------------------------
# VIDEO MODE
# -----------------------------
elif input_source == "Video":
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(temp_path)

        frame_placeholder = st.empty()
        stats_placeholder = st.empty()

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = model(frame_rgb, conf=confidence)[0]
            annotated = draw_boxes(frame_rgb, result, show_conf)

            frame_placeholder.image(annotated, width="stretch")

            frame_count += 1
            fps = frame_count / (time.time() - start_time)

            stats_placeholder.metric("FPS", f"{fps:.2f}")
            stats_placeholder.metric("Frame", frame_count)
            stats_placeholder.metric("Detections", len(result.boxes))

        cap.release()
        st.success("Video processing complete!")