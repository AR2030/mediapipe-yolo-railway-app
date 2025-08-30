# Arabic Sign Language Detection

Real-time Arabic sign language gesture detection using YOLO models within a MediaPipe-based pipeline.

---

## Overview

**Arabic Sign Language Detection** is an application that leverages YOLO (ONNX or PyTorch) models combined with MediaPipe’s hand detection capabilities to perform fast, accurate sign language recognition—specifically for Arabic signs.

---

## Repository Structure

- **Dockerfile** – Container setup for ease of deployment (with optional GPU support).
- **app.py** – Core application: initializes the YOLO model, integrates MediaPipe, and handles live detection.
- **RealTimePrediction.ipynb** – Interactive notebook for testing real-time detection via webcam or video.
- **trainModel.ipynb** – Notebook for training or fine-tuning the YOLO model using custom Arabic sign language datasets.
- **best.onnx**, **best.pt** – Pretrained YOLO model weights (ONNX and PyTorch formats, respectively).
- **arabic_sign_language.json** – Configuration file for classes, detection thresholds, and possibly sign-to-label mapping.
- **freetext.txt** – Placeholder or notes—use it for dataset info, sample lists, or internal documentation.
- **requirements.txt** – Required Python libraries (e.g. MediaPipe, OpenCV, YOLO dependencies).
- **templates/** – Optional web UI components like HTML templates if a GUI is included.

---

## Getting Started

### Prerequisites

- Python 3.7+
- Docker (optional, recommended for containerized environments)

### Installation

#### Option 1: Local Setup

```bash
git clone https://github.com/AR2030/arabic-sign-language-detection.git
cd arabic-sign-language-detection
pip install -r requirements.txt
```
#### Option 2: Docker
```bash
docker build -t arabic-sign-language-detection .
docker run --gpus all -it --rm arabic-sign-language-detection
```
Usage
Running the Application: app.py
``` bash
python app.py
```
This launches the real-time detection pipeline: YOLO model + MediaPipe hand tracking, displaying recognized Arabic signs on webcam or video input.

### Real-Time Testing (Notebook)
* Open RealTimePrediction.ipynb in Jupyter to run live predictions interactively—great for demos and visualization.

### Training Your Own Model
* Run trainModel.ipynb to fine-tune or train your own YOLO model for custom Arabic sign gestures or datasets.

### Configuration
* arabic_sign_language.json: Customize detection classes, confidence thresholds, and label mapping.

* templates/: If applicable, modify the UI templates to suit your branding, language, or functionality needs.

This project serves as a strong foundation for Arabic sign language gesture detection using YOLO models optimized by MediaPipe. It supports training, real-time inference, configuration, and deployment—making it valuable for accessibility, education, and inclusive tech.

## How It Works

The **Arabic Sign Language Detection** system combines **YOLO object detection** with **MediaPipe** to recognize gestures in real time. Here’s the step-by-step pipeline:

1. **Input Capture**
   - The system takes input from a webcam, video file, or image.
   - Each frame is processed individually.

2. **Preprocessing**
   - The frame is resized and normalized to fit the YOLO model input requirements.
   - MediaPipe is optionally used to detect hands and focus only on regions of interest.

3. **YOLO Inference**
   - A pretrained YOLO model (`best.pt` or `best.onnx`) runs on the frame.
   - YOLO outputs bounding boxes, class labels (Arabic signs), and confidence scores.

4. **Postprocessing**
   - Bounding boxes are drawn on the frame.
   - The predicted class (e.g., the Arabic sign label) is overlaid as text.
   - Low-confidence detections are filtered out based on thresholds defined in `arabic_sign_language.json`.

5. **Visualization**
   - The processed frame is displayed in real time with bounding boxes and labels.
   - In Jupyter notebooks, frames can also be displayed inline for testing.

6. **Training (Optional)**
   - Using `trainModel.ipynb`, you can retrain the YOLO model on your own Arabic sign language dataset.
   - After training, export the model to `.pt` (PyTorch) or `.onnx` (ONNX Runtime) for inference in the app.

7. **Deployment**
   - The system can be run locally with Python or containerized with Docker.
   - A lightweight web interface (via Flask + templates) can be used for demonstrations if desired.

---

### Simplified Flowchart
Camera / Video → Preprocessing → YOLO Model → Predictions (bounding boxes + labels) → Display Results

## Dataset & Classes
The YOLO model is trained to recognize the following **Arabic Sign Language gestures**:

- Baba (بابا / Father)
- Book (كتاب)
- Company (شركة)
- Friend (صديق)
- Grandfather (جد)
- Mall (مول / Shopping Center)
- Mama (ماما / Mother)
- Melon (شمام)
- Mosque (مسجد)
- Photographer (مصور)
- Salaam Alaikum (السلام عليكم / Peace be upon you)
- salam (سلام)
- School (مدرسة)
- University (جامعة)

