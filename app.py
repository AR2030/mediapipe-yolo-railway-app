import os
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
app = Flask(__name__)

# --- Load Models on Startup ---
# This section runs only once when the server starts.
MODEL_PATH = "best.pt"
model = None
hands = None
mp_drawing = None

try:
    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"✅ Model loaded successfully. Classes: {class_names}")

    print("Initializing MediaPipe...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3 # Lowered for better detection on varied images
    )
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe initialized successfully.")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Exit if models fail to load
    exit()

# --- Helper Function for Prediction ---
def predict_from_image(image_bytes):
    """Takes image bytes, performs prediction, and returns results."""
    try:
        # Read the image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h, w, _ = cv_image.shape
        
        # Process with MediaPipe
        results_mp = hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        if results_mp.multi_hand_landmarks:
            # Create the white skeleton image for YOLO
            skeleton_image = np.ones((h, w, 3), dtype=np.uint8) * 255
            for hand_landmarks in results_mp.multi_hand_landmarks:
                mp_drawing.draw_landmarks(skeleton_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Predict with YOLOv8 classification model
            results_yolo = model.predict(skeleton_image, verbose=False)
            probs = results_yolo[0].probs
            
            predicted_class = model.names[probs.top1]
            confidence = probs.top1conf.item()
            
            return {
                "success": True,
                "prediction": predicted_class,
                "confidence": f"{confidence:.2%}"
            }
        else:
            return {"success": False, "error": "No hand was detected in the image."}

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"success": False, "error": "An error occurred during prediction."}

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Handle image upload and return prediction as JSON."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if file:
        image_bytes = file.read()
        result = predict_from_image(image_bytes)
        return jsonify(result)
        
# --- Run the App ---
if __name__ == '__main__':
    # Railway provides the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)