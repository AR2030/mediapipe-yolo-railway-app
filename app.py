import os
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import io
import traceback
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Global variables ---
onnx_session = None
hands_detector = None
mp_drawing_connections = None
# MAKE SURE THIS MATCHES YOUR MODEL'S OUTPUTS
CLASS_NAMES = {0: 'baba', 1: 'book', 2: 'company', 3: 'grandfather', 4: 'mall', 5: 'mama', 6: 'melon', 7: 'mosque', 8: 'photograph', 9: 'salaam alaikum', 10: 'salam', 11: 'school', 12: 'university'}
MODEL_INPUT_SIZE = 224

# --- Load Models on Startup ---
try:
    print("Loading ONNX model...")
    onnx_session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
    MODEL_INPUT_SIZE = onnx_session.get_inputs()[0].shape[2] 
    print(f"✅ ONNX model loaded. Input size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
    print(f"Model expects input name: '{onnx_session.get_inputs()[0].name}'")
    print(f"Model has output name: '{onnx_session.get_outputs()[0].name}'")
    print(f"Model has {len(CLASS_NAMES)} classes defined in API.")


    print("Initializing MediaPipe...")
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)
    mp_drawing_connections = mp_hands.HAND_CONNECTIONS
    print("✅ MediaPipe initialized.")

except Exception as e:
    print(f"❌ Error loading models: {e}")

# --- Helper Functions (with FIXES) ---

def normalize_landmarks(landmarks):
    landmarks_np = np.array(landmarks, dtype=np.float32) # Use float32
    origin = landmarks_np[0].copy()
    landmarks_np -= origin
    max_val = np.max(np.abs(landmarks_np))
    
    # --- FIX 1: Prevent Division by Zero ---
    # Use a small epsilon to avoid dividing by zero if all landmarks are the same
    if max_val > 1e-6:
        landmarks_np /= max_val
        
    return landmarks_np

def draw_landmarks_on_canvas(list_of_hands_landmarks, image_size):
    # This function is likely fine, no changes needed here.
    img = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    for landmarks in list_of_hands_landmarks:
        center_x, center_y = image_size // 2, image_size // 2
        scale = image_size * 0.4 # Slightly increased scale
        points = []
        for lm in landmarks:
            x = int(center_x + lm[0] * scale)
            y = int(center_y + lm[1] * scale)
            points.append((x, y))
        if mp_drawing_connections:
            for connection in mp_drawing_connections:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(img, points[start_idx], points[end_idx], (0, 0, 255), 2)
        for point in points:
            cv2.circle(img, point, 4, (255, 0, 0), -1)
    return img

# --- ONNX Preprocessing and Prediction Logic (with DEBUGGING) ---

def preprocess_for_onnx(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # --- FIX 2: Ensure dtype is float32 from the start ---
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    return np.expand_dims(image_chw, axis=0)

def predict_from_image_bytes(image_bytes):
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        results_mp = hands_detector.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        if not results_mp.multi_hand_landmarks:
            return {"success": False, "error": "No hand was detected in the image."}

        all_hands_landmarks = []
        for hand_world_landmarks in results_mp.multi_hand_landmarks:
            # --- FIX 3: Using a more stable set of landmarks ---
            # Using 'hand_world_landmarks' can sometimes be more stable if available,
            # but let's stick to the screen coordinates for consistency with your training.
            # We will just extract the x,y,z from the standard landmarks.
            hand_screen_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_world_landmarks.landmark]
            
            # --- FIX 4: Check if landmark list is empty before normalizing ---
            if not hand_screen_landmarks:
                continue # Skip this hand if it has no landmarks
            
            normalized_lms = normalize_landmarks(hand_screen_landmarks)
            all_hands_landmarks.append(normalized_lms)
            
        if not all_hands_landmarks:
             return {"success": False, "error": "Hand detected, but landmarks could not be processed."}


        skeleton_image = draw_landmarks_on_canvas(all_hands_landmarks, MODEL_INPUT_SIZE)
        input_tensor = preprocess_for_onnx(skeleton_image)
        
        # --- DEBUGGING: Print shape and type right before inference ---
        print(f"DEBUG: Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        
        # This is where the error likely happens
        onnx_outputs = onnx_session.run([output_name], {input_name: input_tensor})
        
        scores = onnx_outputs[0][0]
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        
        top1_index = np.argmax(probs)
        confidence = probs[top1_index]
        predicted_class = CLASS_NAMES.get(top1_index, "Unknown Class")
            
        return {"success": True, "prediction": predicted_class, "confidence": f"{confidence:.2%}"}

    except Exception as e:
        print(f"--- PREDICTION FAILED ---")
        # This will print the full error to your console
        traceback.print_exc()
        print(f"--- END OF ERROR ---")
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
        result = predict_from_image_bytes(image_bytes)
        return jsonify(result)
        
# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
