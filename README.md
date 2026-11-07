# Space-Objects-Around-Black-Holes
Deep Learning based astronomical image classification and object detection using YOLO26
# 🌌 Space Objects Around Black Holes

A Deep Learning project built using Python, YOLO26, Streamlit, and computer vision to classify astronomical images and detect black holes, galaxies, jets, nebulae, and stars.

## Features

- 15-class astronomical image classification
- Five-class astronomical object detection
- YOLO26-based trained deep learning models
- Image upload and prediction through Streamlit
- Top-3 classification predictions with confidence scores
- Object detection with bounding boxes and confidence scores
- Detection results displayed in a table
- Download annotated detection images
- Interactive web-based application

## Technologies

- Python
- YOLO26
- Ultralytics
- Streamlit
- OpenCV
- NumPy
- Pandas
- Pillow
- PyTorch

## Dataset

### Classification Dataset
- 15 astronomical object classes
- Based on the Space Objects Around Black Holes (ESA) Kaggle dataset.

### Object Detection Dataset

| Dataset subset | Images |

| Training | 329 |
| Validation | 66 |
| Testing | 44 |
| **Total** | **439** |

### Detection Classes

1. Blackhole
2. Galaxy
3. Jet
4. Nebula
5. Star

## Model Evaluation

The object detection model is evaluated using:

- Precision
- Recall
- Mean Average Precision at 50% IoU (mAP@50)
- Mean Average Precision at 50–95% IoU (mAP@50–95)

The classification model is evaluated using Top-1 and Top-5 accuracy.

 Run Locally

 1. Install Python

Install Python and ensure it is available from your terminal.

 2. Download the project

Clone the repository:

## Model Files

The application requires two trained model checkpoints:

- Classification model: `yolo26_results/yolo26_finetune_320/weights/best.pt`
- Detection model:`runs/five_class_detector_v5_cpu/weights/best.pt`

Make sure both files are present in their respective folders before running the application.

## Project Structure

text
Space-Objects-Around-Black-Holes/
│
├── app.py
├── requirements.txt
├── README.md
│
├── yolo26_results/
│   └── yolo26_finetune_320/
│       └── weights/
│           └── best.pt
│
├── runs/
│   └── five_class_detector_v5_cpu/
│       └── weights/
│           └── best.pt
│
├── blackhole_images_combined/
│
├── classification/
│
├── detection/
│
└── dataset_info/


## Important Note

This project is developed for educational and research purposes. Model predictions are based on learned image patterns and should not be interpreted as verified astronomical discoveries or scientific measurements.

Detection bounding boxes and classification confidence scores are model predictions and may contain errors.

## Project Links

GitHub Repository:

https://github.com/jasminigollapati77-alt/Space-Objects-Around-Black-Holes

Project Demonstration Video:

https://youtu.be/39BzfIE5few
