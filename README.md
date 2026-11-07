# Space Objects Around Black Holes

## Project Overview

A deep learning project for astronomical image classification
and object detection using YOLO26 and Streamlit.

## Application

Run the application from this project directory:

    pip install -r requirements.txt
    streamlit run app.py

The application uses two separate trained models.

## Classification

15 astronomical image categories.

Model:
yolo26_results/yolo26_finetune_320/weights/best.pt

Previously measured held-out results:
Top-1 accuracy: 86.21% (25/29)
Top-5 accuracy: 100.00% (29/29)

## Object Detection

Five categories:
blackhole, galaxy, jet, nebula and star.

Model:
runs/five_class_detector_v5_cpu/weights/best.pt

Dataset:
blackhole_images_combined/

Train: 329 images
Validation: 66 images
Test: 44 images
Total: 439 images

## Detection Test Results

Precision: 65.08%
Recall: 44.35%
mAP50: 43.46%
mAP50-95: 27.54%

The detector metrics and classifier accuracy are different
evaluation measures for different tasks.

## Important Reproducibility Note

The detector training configuration references
/content/dataset_v5/data.yaml. That original training path
was not available during final packaging.

The detector was separately evaluated on the included
blackhole_images_combined test split.

The available original application source, configurations,
model weights and evaluation outputs are included.

Original standalone training notebooks/scripts were not
found in the inspected project inventory. The configuration
files are not substitutes for those original scripts.

## Dataset

The complete 439-image object-detection dataset is included,
with its train, valid and test image and label folders.
