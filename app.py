import os
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import io
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
app = Flask(__name__)

# --- Global variables ---
onnx_session = None
hands_detector = None # Renamed to be more descriptive
mp_drawing_connections = None
CLASS_NAMES = {0: 'baba', 1: 'book', 2: 'company', 3: 'grandfather', 4: 'mall', 5: 'mama', 6: 'melon', 7: 'mosque', 8: 'photograph', 9: 'salaam alaikum', 10: 'salam', 11: 'school', 12: 'university'}
MODEL_INPUT_SIZE = 224 # Define a default, will be updated from model

# --- Load Models on Startup ---
try:
    print("Loading ONNX model...")
    onnx_session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
    # Get the input size from the ONNX model itself for robustness
    MODEL_INPUT_SIZE = onnx_session.get_inputs()[0].shape[2] 
    print(f"✅ ONNX model loaded successfully. Model expects input size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")

    print("Initializing MediaPipe...")
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3 # Slightly higher confidence for better results
    )
    mp_drawing_connections = mp_hands.HAND_CONNECTIONS
    print("✅ MediaPipe initialized successfully.")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    # exit() # Comment out exit for easier debugging in some environments

# --- Helper Functions (The Correct Preprocessing Pipeline) ---

def normalize_landmarks(landmarks):
    """Normalizes landmarks to be centered and scale-invariant."""
    landmarks_np = np.array(landmarks)
    origin = landmarks_np[0].copy() # Treat wrist as the origin (0,0)
    landmarks_np -= origin
    max_val = np.max(np.abs(landmarks_np))
    if max_val > 0:
        landmarks_np /= max_val
    return landmarks_np

def draw_landmarks_on_canvas(list_of_hands_landmarks, image_size):
    """Renders normalized landmarks onto a blank, fixed-size canvas."""
    img = np.full((image_size, image_size, 3), 255, dtype=np.uint8) # White background
    
    for landmarks in list_of_hands_landmarks:
        center_x, center_y = image_size // 2, image_size // 2
        # Use a slightly smaller scale to ensure both hands fit comfortably
        scale = image_size * 0.35 
        
        points = []
        for lm in landmarks:
            # Scale normalized landmarks to fit the canvas
            x = int(center_x + lm[0] * scale)
            y = int(center_y + lm[1] * scale)
            points.append((x, y))

        # Draw connections (skeleton) in Red
        if mp_drawing_connections:
            for connection in mp_drawing_connections:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(img, points[start_idx], points[end_idx], (0, 0, 255), 2)
        
        # Draw landmarks (joints) in Blue
        for point in points:
            cv2.circle(img, point, 4, (255, 0, 0), -1)
            
    return img

# --- ONNX Preprocessing and Prediction Logic ---

def preprocess_for_onnx(image_bgr):
    """Prepares the final canvas image for the ONNX model."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    return np.expand_dims(image_chw, axis=0)

def predict_from_image_bytes(image_bytes):
    """
    The main prediction function. Takes image bytes, performs the full
    normalization and rendering pipeline, and returns the prediction.
    """
    try:
        # 1. Load image and find landmarks with MediaPipe
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        results_mp = hands_detector.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        if not results_mp.multi_hand_landmarks:
            return {"success": False, "error": "No hand was detected in the image."}

        # 2. Normalize the landmarks (The CRITICAL step)
        all_hands_landmarks = []
        if len(results_mp.multi_hand_landmarks) == 2:
            # TWO-HAND CASE: Normalize relative to the center of the two hands
            hand1_lms = np.array([[lm.x, lm.y, lm.z] for lm in results_mp.multi_hand_landmarks[0].landmark])
            hand2_lms = np.array([[lm.x, lm.y, lm.z] for lm in results_mp.multi_hand_landmarks[1].landmark])
            center_point = (hand1_lms[0] + hand2_lms[0]) / 2
            combined_lms = np.vstack([hand1_lms, hand2_lms])
            combined_lms -= center_point
            max_val = np.max(np.abs(combined_lms))
            if max_val > 0: combined_lms /= max_val
            all_hands_landmarks.append(combined_lms[:21])
            all_hands_landmarks.append(combined_lms[21:])
        else:
            # ONE-HAND CASE
            hand_lms = [[lm.x, lm.y, lm.z] for lm in results_mp.multi_hand_landmarks[0].landmark]
            normalized_lms = normalize_landmarks(hand_lms)
            all_hands_landmarks.append(normalized_lms)
        
        # 3. Render the normalized landmarks onto a clean, fixed-size canvas
        skeleton_image = draw_landmarks_on_canvas(all_hands_landmarks, MODEL_INPUT_SIZE)
        
        # 4. Preprocess this final canvas for ONNX and run inference
        input_tensor = preprocess_for_onnx(skeleton_image)
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        onnx_outputs = onnx_session.run([output_name], {input_name: input_tensor})
        
        scores = onnx_outputs[0][0]
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        
        top1_index = np.argmax(probs)
        confidence = probs[top1_index]
        
        # Handle case where index might be out of bounds for the dictionary
        predicted_class = CLASS_NAMES.get(top1_index, "Unknown Class")
            
        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2%}"
        }

    except Exception as e:
        # It's good practice to log the actual error on the server
        print(f"Prediction error: {e}") 
        import traceback
        traceback.print_exc()
        return {"success": False, "error": "An internal error occurred during prediction."}

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if file:
        image_bytes = file.read()
        # Use the new function name
        result = predict_from_image_bytes(image_bytes) 
        return jsonify(result)
        
# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
