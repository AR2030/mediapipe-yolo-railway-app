import os
import cv2
import numpy as np
import mediapipe as mp
import torch
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = 'best.pt'         # Path to your trained YOLOv8 model
IMAGE_SIZE = 224              # Input size for the model
CONFIDENCE_THRESHOLD = 0.70   # Minimum confidence to display prediction

# Initialize Flask app
app = Flask(__name__)

# Initialize YOLO model
def load_model(path):
    model = YOLO(path)
    return model, model.names

# Normalize landmarks similar to training pipeline
def normalize_landmarks(landmarks):
    lm = np.array(landmarks)
    origin = lm[0].copy()
    lm -= origin
    m = np.max(np.abs(lm))
    if m > 0:
        lm /= m
    return lm

# Draw landmarks onto a white canvas
def draw_landmarks_canvas(hands_list, size, connections):
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    for landmarks in hands_list:
        cx, cy = size // 2, size // 2
        scale = size * 0.35
        pts = [(int(cx + lm[0] * scale), int(cy + lm[1] * scale)) for lm in landmarks]
        for s, e in connections:
            if s < len(pts) and e < len(pts):
                cv2.line(canvas, pts[s], pts[e], (0, 0, 255), 2)
        for p in pts:
            cv2.circle(canvas, p, 4, (255, 0, 0), -1)
    return canvas

# Process a single image
def process_image(img, model, class_names):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    if img is None:
        return None, "Failed to process image."

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    text = "No Hand Detected"

    if res.multi_hand_landmarks:
        all_hands = []
        if len(res.multi_hand_landmarks) == 2:
            l1 = np.array([[lm.x, lm.y, lm.z] for lm in res.multi_hand_landmarks[0].landmark])
            l2 = np.array([[lm.x, lm.y, lm.z] for lm in res.multi_hand_landmarks[1].landmark])
            center = (l1[0] + l2[0]) / 2
            combined = np.vstack([l1, l2]) - center
            m = np.max(np.abs(combined))
            if m > 0:
                combined /= m
            all_hands.append(combined[:21])
            all_hands.append(combined[21:])
        else:
            l = [[lm.x, lm.y, lm.z] for lm in res.multi_hand_landmarks[0].landmark]
            all_hands.append(normalize_landmarks(l))

        canvas = draw_landmarks_canvas(
            all_hands,
            IMAGE_SIZE,
            mp.solutions.hands.HAND_CONNECTIONS
        )

        # YOLO prediction
        yres = model.predict(source=canvas, verbose=False)[0]
        probs = yres.probs.data
        conf = float(torch.max(probs))
        if conf > CONFIDENCE_THRESHOLD:
            cls = int(torch.argmax(probs))
            name = class_names[cls]
            text = f"{name} ({conf:.2%})"
        else:
            text = "Uncertain"
    else:
        canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)

    return canvas, text

@app.route('/')
def home():
    return render_template('index.html')

# API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400

    # Read the uploaded image into memory
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'success': False, 'error': 'Invalid image format.'}), 400

    # Load YOLO model
    model, class_names = load_model(MODEL_PATH)

    # Process the image and get the prediction
    canvas, result_text = process_image(img, model, class_names)

    if canvas is None:
        return jsonify({'success': False, 'error': result_text}), 500

    # Return prediction result
    return jsonify({
        'success': True,
        'result': result_text
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
