
from pathlib import Path
from io import BytesIO

import streamlit as st
import numpy as np
import pandas as pd
import cv2

from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Space Objects Around Black Holes",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Space Objects Around Black Holes")

st.write(
    "Classify astronomical images and detect black holes, galaxies, "
    "jets, nebulae and stars using trained YOLO models."
)


# ============================================================
# 2. MODEL PATHS
# ============================================================

# Automatically locate the folder containing app.py
BASE_DIR = Path(__file__).resolve().parent

# 15-class YOLO26 classifier
CLASSIFIER_PATH = (
    BASE_DIR
    / "yolo26_results"
    / "yolo26_finetune_320"
    / "weights"
    / "best.pt"
)

# 5-class YOLO26 object detector
DETECTOR_PATH = (
    BASE_DIR
    / "runs"
    / "five_class_detector_v5_cpu"
    / "weights"
    / "best.pt"
)


# ============================================================
# 3. LOAD TRAINED YOLO MODELS
# ============================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# Check classifier file
if not CLASSIFIER_PATH.is_file():
    st.error(
        f"Classification model not found: {CLASSIFIER_PATH}"
    )
    st.stop()


# Check detector file
if not DETECTOR_PATH.is_file():
    st.error(
        f"Detection model not found: {DETECTOR_PATH}"
    )
    st.stop()


# Load both models
try:

    with st.spinner("Loading trained YOLO models..."):

        classifier = load_model(CLASSIFIER_PATH)
        detector = load_model(DETECTOR_PATH)

except Exception as exc:

    st.error("Unable to load the trained YOLO models.")
    st.exception(exc)
    st.stop()


st.success("Both trained YOLO models loaded successfully.")


# ============================================================
# 4. PREVIOUSLY REPORTED CLASSIFICATION TEST RESULTS
# ============================================================

st.divider()

st.header("🏆 15-Class Classification Test Results")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Top-1 Accuracy",
        "72.41%"
    )

    st.caption(
        "21 / 29 test images correctly classified"
    )


with col2:

    st.metric(
        "Top-5 Accuracy",
        "96.55%"
    )

    st.caption(
        "28 / 29 test images included in top five"
    )


st.info(
    "These are the previously reported held-out test results. "
    "They are historical evaluation metrics, not the confidence "
    "of an uploaded image. They are not recalculated automatically "
    "when this application starts."
)


# ============================================================
# 5. UPLOAD ONE ASTRONOMICAL IMAGE
# ============================================================

st.divider()

st.header("🖼️ Upload an Astronomical Image")

uploaded_file = st.file_uploader(
    "Choose one JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
    key="current_astronomy_image"
)


if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Your Uploaded Image")

        st.image(
            image,
            width="stretch"
        )


        # ====================================================
        # 6. 15-CLASS IMAGE CLASSIFICATION
        # ====================================================

        st.divider()

        st.header("🔭 15-Class Image Classification")

        with st.spinner("Classifying the uploaded image..."):

            classification_result = classifier.predict(
                source=image,
                imgsz=320,
                verbose=False
            )[0]


        if classification_result.probs is not None:

            probabilities = (
                classification_result.probs.data.cpu().numpy()
            )

            class_names = classification_result.names

            top_indices = np.argsort(probabilities)[::-1][:3]

            st.subheader("Top 3 Predicted Classes")

            for rank, class_index in enumerate(
                top_indices,
                start=1
            ):

                class_name = class_names[int(class_index)]

                confidence = float(
                    probabilities[class_index]
                )

                st.write(
                    f"**{rank}. {class_name}** — "
                    f"{confidence * 100:.2f}%"
                )

                st.progress(
                    min(max(confidence, 0.0), 1.0)
                )


            st.caption(
                "Confidence represents the model's prediction score. "
                "It does not guarantee that the prediction is correct."
            )

        else:

            st.warning(
                "The classification model did not return "
                "classification probabilities."
            )


        # ====================================================
        # 7. FIVE-CLASS OBJECT DETECTION
        # ====================================================

        st.divider()

        st.header("🎯 Roboflow Object Detection")

        st.write(
            "Boxes and labels are generated by the trained "
            "five-class object detection model."
        )


        confidence_threshold = st.slider(
            "Detection confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.20,
            step=0.05,
            help=(
                "Lower values may show more predictions, including "
                "incorrect ones. Higher values show fewer predictions."
            )
        )


        with st.spinner("Detecting objects in your image..."):

            detection_result = detector.predict(
                source=image,
                conf=confidence_threshold,
                verbose=False
            )[0]


        # Generate annotated image
        annotated_bgr = detection_result.plot(
            labels=True,
            conf=True,
            boxes=True
        )

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB
        )

        annotated_image = Image.fromarray(
            annotated_rgb
        )


        # Display original and detected images
        left, right = st.columns(2)

        with left:

            st.subheader("Original Image")

            st.image(
                image,
                width="stretch"
            )


        with right:

            st.subheader("Detected Objects")

            st.image(
                annotated_image,
                width="stretch"
            )


        # ====================================================
        # 8. DETECTION DETAILS
        # ====================================================

        boxes = detection_result.boxes

        if boxes is not None and len(boxes) > 0:

            rows = []

            coordinates = boxes.xyxy.cpu().numpy()

            confidences = boxes.conf.cpu().numpy()

            class_ids = (
                boxes.cls.cpu().numpy().astype(int)
            )

            detection_names = detection_result.names


            for index, (xyxy, score, class_id) in enumerate(
                zip(coordinates, confidences, class_ids),
                start=1
            ):

                x1, y1, x2, y2 = xyxy

                rows.append({

                    "Object": index,

                    "Predicted Class": (
                        detection_names[class_id]
                    ),

                    "Confidence (%)": round(
                        float(score) * 100,
                        2
                    ),

                    "X1": round(float(x1), 1),

                    "Y1": round(float(y1), 1),

                    "X2": round(float(x2), 1),

                    "Y2": round(float(y2), 1)

                })


            st.metric(
                "Objects Detected",
                len(rows)
            )


            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True
            )


        else:

            st.warning(
                "No objects were detected at this confidence threshold. "
                "Try lowering the threshold or use an image containing "
                "objects similar to the detector's training data."
            )


        # ====================================================
        # 9. DOWNLOAD DETECTED IMAGE
        # ====================================================

        buffer = BytesIO()

        annotated_image.save(
            buffer,
            format="PNG"
        )


        st.download_button(
            label="⬇️ Download Detected Image",

            data=buffer.getvalue(),

            file_name="roboflow_detected_image.png",

            mime="image/png"
        )


        st.caption(
            "Detection boxes are model predictions, "
            "not guaranteed ground-truth annotations."
        )


    except Exception as exc:

        st.error(
            "An error occurred while processing this image."
        )

        st.exception(exc)


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.caption(
    "Space Objects Around Black Holes | "
    "YOLO26 Classification + Roboflow Object Detection"
)
