from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import time
from ultralytics import YOLO
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- MODELS ----------
try:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    logger.info("FaceMesh model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load FaceMesh: {e}")
    raise

try:
    yolo = YOLO("yolov8n.pt")  # phone detection
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise

# ---------- LANDMARKS ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# ---------- SESSION STATE (single user demo) ----------
# Note: For production, use a proper session management system
state = {
    "eyes_closed_since": None,
    "last_face_seen": time.time(),
    "attention_frames": 0,
    "total_frames": 0,
    "phone_hits": 0,
    "cheat_score": 0
}

# ---------- UTILS ----------
def eye_ratio(eye):
    """Calculate eye aspect ratio"""
    try:
        v1 = np.linalg.norm(eye[1] - eye[5])
        v2 = np.linalg.norm(eye[2] - eye[4])
        h = np.linalg.norm(eye[0] - eye[3])
        if h == 0:
            return 0.3  # Default value to avoid division by zero
        return (v1 + v2) / (2 * h)
    except Exception as e:
        logger.warning(f"Error calculating eye ratio: {e}")
        return 0.3

def mouth_ratio(m):
    """Calculate mouth aspect ratio"""
    try:
        v = np.linalg.norm(m[0] - m[1])
        h = np.linalg.norm(m[2] - m[3])
        if h == 0:
            return 0.3  # Default value to avoid division by zero
        return v / h
    except Exception as e:
        logger.warning(f"Error calculating mouth ratio: {e}")
        return 0.3

# ---------- ROUTES ----------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Camera Monitor API is running!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode image
        img_bytes = base64.b64decode(data["image"])
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        response = {
            "sleepy": False,
            "face_missing": False,
            "multiple_faces": False,
            "looking_away": False,
            "yawning": False,
            "phone_detected": False,
            "attention": 0,
            "cheating": False,
            "cheat_score": 0
        }

        # Process face mesh
        res = face_mesh.process(rgb)
        state["total_frames"] += 1

        # ---------- FACE ----------
        if not res.multi_face_landmarks:
            if time.time() - state["last_face_seen"] > 3:
                response["face_missing"] = True
                state["cheat_score"] += 2
            return finalize(response)

        state["last_face_seen"] = time.time()

        if len(res.multi_face_landmarks) > 1:
            response["multiple_faces"] = True
            state["cheat_score"] += 5

        face = res.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array with error handling
        try:
            lm = np.array([[int(l.x * w), int(l.y * h)] for l in face.landmark])
        except Exception as e:
            logger.warning(f"Error processing landmarks: {e}")
            return finalize(response)

        # ---------- SLEEP ----------
        try:
            ear = (eye_ratio(lm[LEFT_EYE]) + eye_ratio(lm[RIGHT_EYE])) / 2
            if ear < 0.25:
                if not state["eyes_closed_since"]:
                    state["eyes_closed_since"] = time.time()
                elif time.time() - state["eyes_closed_since"] > 4:
                    response["sleepy"] = True
                    state["cheat_score"] += 2
            else:
                state["eyes_closed_since"] = None
        except Exception as e:
            logger.warning(f"Error calculating eye aspect ratio: {e}")

        # ---------- YAWN ----------
        try:
            if mouth_ratio(lm[MOUTH]) > 0.65:
                response["yawning"] = True
        except Exception as e:
            logger.warning(f"Error calculating mouth ratio: {e}")

        # ---------- ATTENTION ----------
        try:
            nose_x = lm[1][0]
            if w * 0.35 < nose_x < w * 0.65:
                state["attention_frames"] += 1
            else:
                response["looking_away"] = True
                state["cheat_score"] += 1

            # Calculate attention percentage, avoid division by zero
            if state["total_frames"] > 0:
                response["attention"] = int(
                    (state["attention_frames"] / state["total_frames"]) * 100
                )
            else:
                response["attention"] = 0
        except Exception as e:
            logger.warning(f"Error calculating attention: {e}")

        # ---------- PHONE (YOLO) ----------
        try:
            results = yolo(img, conf=0.4, verbose=False)[0]
            if hasattr(results, 'boxes') and results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if yolo.names[cls] == "cell phone":
                        response["phone_detected"] = True
                        state["phone_hits"] += 1
                        state["cheat_score"] += 5
                        break
        except Exception as e:
            logger.warning(f"Error in YOLO detection: {e}")

        return finalize(response)
    
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# ---------- HARDENING ----------
def finalize(response):
    """Finalize response with cheat score and cheating detection"""
    # Apply decay to cheat score
    state["cheat_score"] = max(0, state["cheat_score"] - 0.1)
    response["cheat_score"] = int(state["cheat_score"])

    # Check for cheating conditions
    if state["cheat_score"] >= 10 or state["phone_hits"] >= 2:
        response["cheating"] = True

    # Add timestamp
    response["timestamp"] = time.time()
    
    return jsonify(response)

@app.route("/reset", methods=["POST"])
def reset_state():
    """Reset the monitoring state (useful for testing)"""
    global state
    state = {
        "eyes_closed_since": None,
        "last_face_seen": time.time(),
        "attention_frames": 0,
        "total_frames": 0,
        "phone_hits": 0,
        "cheat_score": 0
    }
    return jsonify({"message": "State reset successfully", "state": state})

@app.route("/status", methods=["GET"])
def get_status():
    """Get current monitoring status"""
    return jsonify({
        "state": state,
        "models_loaded": True,
        "server_time": time.time()
    })

# ---------- RUN ----------
if __name__ == "__main__":
    # Validate models are loaded
    try:
        logger.info("Starting Flask server...")
        logger.info(f"YOLO classes available: {list(yolo.names.values())[:5]}...")
        
        app.run(
            host="0.0.0.0", 
            port=8000, 
            debug=False,
            threaded=True  # Enable threading for concurrent requests
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import mediapipe as mp
# import numpy as np
# import base64
# import time
# from ultralytics import YOLO
# from collections import deque

# # ================== CONFIG ==================
# TEST_MODE = False        # <<< SET FALSE WHEN USING NEXT.JS
# CAMERA_INDEX = 0

# EAR_THRESHOLD = 0.23
# EYE_CLOSE_TIME = 3.0

# MAR_THRESHOLD = 0.65
# YAWN_FRAMES = 8

# HEAD_YAW_LIMIT = 25     

# PHONE_CONF = 0.45
# PHONE_SKIP_FRAMES = 10

# # ============================================

# app = Flask(__name__)
# CORS(app)

# # ---------- MODELS ----------
# face_mesh = mp.solutions.face_mesh.FaceMesh(
#     max_num_faces=2,
#     refine_landmarks=True
# )
# yolo = YOLO("yolov8n.pt")

# # ---------- LANDMARKS ----------
# LEFT_EYE = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# MOUTH = [13, 14, 78, 308]

# # ---------- STATE ----------
# state = {
#     "eyes_closed_since": None,
#     "yawn_buffer": deque(maxlen=YAWN_FRAMES),
#     "attention_frames": 0,
#     "total_frames": 0,
#     "phone_hits": 0,
#     "cheat_score": 0,
#     "frame_count": 0
# }

# # ---------- UTILS ----------
# def eye_ratio(eye):
#     v1 = np.linalg.norm(eye[1] - eye[5])
#     v2 = np.linalg.norm(eye[2] - eye[4])
#     h = np.linalg.norm(eye[0] - eye[3])
#     return (v1 + v2) / (2 * h)

# def mouth_ratio(m):
#     return np.linalg.norm(m[0] - m[1]) / np.linalg.norm(m[2] - m[3])

# def head_yaw(landmarks, w):
#     left = landmarks[234][0]
#     right = landmarks[454][0]
#     center = (left + right) / 2
#     yaw = ((center - w/2) / (w/2)) * 45
#     return yaw

# # ---------- CORE ANALYSIS ----------
# def analyze_frame(img):
#     h, w, _ = img.shape
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     result = {
#         "sleepy": False,
#         "looking_away": False,
#         "yawning": False,
#         "phone_detected": False,
#         "cheating": False
#     }

#     state["total_frames"] += 1
#     state["frame_count"] += 1

#     res = face_mesh.process(rgb)
#     if not res.multi_face_landmarks:
#         state["cheat_score"] += 1
#         return result

#     face = res.multi_face_landmarks[0]
#     lm = np.array([[int(l.x*w), int(l.y*h)] for l in face.landmark])

#     # ---------- SLEEP ----------
#     ear = (eye_ratio(lm[LEFT_EYE]) + eye_ratio(lm[RIGHT_EYE])) / 2
#     if ear < EAR_THRESHOLD:
#         if not state["eyes_closed_since"]:
#             state["eyes_closed_since"] = time.time()
#         elif time.time() - state["eyes_closed_since"] > EYE_CLOSE_TIME:
#             result["sleepy"] = True
#             state["cheat_score"] += 2
#     else:
#         state["eyes_closed_since"] = None

#     # ---------- YAWN ----------
#     mar = mouth_ratio(lm[MOUTH])
#     state["yawn_buffer"].append(mar)
#     if sum(m > MAR_THRESHOLD for m in state["yawn_buffer"]) > YAWN_FRAMES // 2:
#         result["yawning"] = True

#     # ---------- HEAD POSE ----------
#     yaw = head_yaw(lm, w)
#     if abs(yaw) > HEAD_YAW_LIMIT:
#         result["looking_away"] = True
#         state["cheat_score"] += 1
#     else:
#         state["attention_frames"] += 1

#     # ---------- PHONE (OPTIMIZED) ----------
#     if state["frame_count"] % PHONE_SKIP_FRAMES == 0:
#         yolo_res = yolo(img, conf=PHONE_CONF, verbose=False)[0]
#         for box in yolo_res.boxes:
#             cls = int(box.cls[0])
#             if yolo.names[cls] == "cell phone":
#                 result["phone_detected"] = True
#                 state["phone_hits"] += 1
#                 state["cheat_score"] += 5
#                 break

#     if state["cheat_score"] >= 10 or state["phone_hits"] >= 2:
#         result["cheating"] = True

#     return result

# # ---------- FLASK ----------
# @app.route("/analyze", methods=["POST"])
# def analyze_api():
#     data = request.json
#     img_bytes = base64.b64decode(data["image"])
#     img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
#     return jsonify(analyze_frame(img))

# # ---------- TEST MODE ----------
# def run_test_camera():
#     cap = cv2.VideoCapture(CAMERA_INDEX)
#     print("TEST MODE: Press ESC to exit")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         res = analyze_frame(frame)

#         y = 30
#         for k, v in res.items():
#             if v:
#                 cv2.putText(frame, f"{k.upper()}", (20, y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
#                 y += 30

#         cv2.imshow("AI Proctor Test", frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # ---------- RUN ----------
# if __name__ == "__main__":
#     if TEST_MODE:
#         run_test_camera()
#     else:
#         app.run(host="0.0.0.0", port=8000, debug=False)





# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import csv
# from datetime import datetime

# print("Siya Ram")

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Camera error")
#     exit()

# # ---------------- MEDIAPIPE ----------------
# mp_face = mp.solutions.face_mesh
# face_mesh = mp_face.FaceMesh(
#     max_num_faces=3,
#     refine_landmarks=True
# )

# # ---------------- LANDMARKS ----------------
# LEFT_EYE = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# MOUTH = [13, 14, 78, 308]  # upper, lower, left, right

# # ---------------- UTIL FUNCTIONS ----------------
# def eye_ratio(eye):
#     v1 = np.linalg.norm(eye[1] - eye[5])
#     v2 = np.linalg.norm(eye[2] - eye[4])
#     h = np.linalg.norm(eye[0] - eye[3])
#     return (v1 + v2) / (2 * h)

# def mouth_ratio(m):
#     vertical = np.linalg.norm(m[0] - m[1])
#     horizontal = np.linalg.norm(m[2] - m[3])
#     return vertical / horizontal

# # ---------------- METRICS ----------------
# eyes_closed_start = None
# total_sleep_time = 0
# attention_score = 0
# frames = 0
# yawn_count = 0

# # ---------------- LOG FILE ----------------
# log_file = open("session_log.csv", "w", newline="", encoding="utf-8")
# writer = csv.writer(log_file)
# writer.writerow(["time", "status", "attention", "faces", "yawns"])

# print("Monitoring started — ESC to exit")

# # ---------------- MAIN LOOP ----------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = face_mesh.process(rgb)

#     status = "Active"
#     faces_detected = 0

#     if result.multi_face_landmarks:
#         faces_detected = len(result.multi_face_landmarks)

#         # ❌ CHEATING: MULTIPLE FACES
#         if faces_detected > 1:
#             status = "Cheating: Multiple Faces"
#             cv2.putText(frame, status, (30, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

#         face = result.multi_face_landmarks[0]
#         landmarks = np.array([[int(l.x*w), int(l.y*h)] for l in face.landmark])

#         # -------- EYES --------
#         left_eye = landmarks[LEFT_EYE]
#         right_eye = landmarks[RIGHT_EYE]
#         ear = (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2

#         # TIME-BASED DROWSINESS
#         if ear < 0.25:
#             if eyes_closed_start is None:
#                 eyes_closed_start = time.time()
#         else:
#             if eyes_closed_start:
#                 total_sleep_time += time.time() - eyes_closed_start
#                 eyes_closed_start = None

#         if total_sleep_time > 5:
#             status = "Drowsy 😴"

#         # -------- YAWN --------
#         mouth = landmarks[MOUTH]
#         mar = mouth_ratio(mouth)
#         if mar > 0.6:
#             yawn_count += 1
#             status = "Yawning 😮"

#         # -------- ATTENTION --------
#         nose_x = landmarks[1][0]
#         if w*0.4 < nose_x < w*0.6:
#             attention_score += 1

#         frames += 1

#     else:
#         status = "Not Present ❌"

#     attention_percent = int((attention_score / max(frames,1)) * 100)

#     # -------- DISPLAY --------
#     cv2.putText(frame, f"Status: {status}", (20,80),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
#     cv2.putText(frame, f"Attention: {attention_percent}%", (20,120),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

#     cv2.imshow("AI Proctor", frame)

#     # -------- LOG --------
#     writer.writerow([
#         datetime.now().isoformat(),
#         status,
#         attention_percent,
#         faces_detected,
#         yawn_count
#     ])

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# log_file.close()
# cv2.destroyAllWindows()
# print("Monitoring stopped")


# print("siya Ram")

# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# # Initialize camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not access camera")
#     exit()


# mp_face = mp.solutions.face_mesh
# face_mesh = mp_face.FaceMesh(max_num_faces=1)

# mp_draw = mp.solutions.drawing_utils

# # Eye landmarks (Mediapipe)
# LEFT_EYE = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# # Eye aspect ratio
# def eye_ratio(eye):
#     vertical1 = np.linalg.norm(eye[1] - eye[5])
#     vertical2 = np.linalg.norm(eye[2] - eye[4])
#     horizontal = np.linalg.norm(eye[0] - eye[3])
#     return (vertical1 + vertical2) / (2.0 * horizontal)

# sleep_counter = 0
# inactive_start = time.time()

# print("Starting behavior monitor... Press ESC to exit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     status = "Active"

#     if results.multi_face_landmarks:
#         inactive_start = time.time()
#         face = results.multi_face_landmarks[0]

#         h, w, _ = frame.shape
#         landmarks = np.array([
#             [int(l.x * w), int(l.y * h)] for l in face.landmark
#         ])

#         left_eye = landmarks[LEFT_EYE]
#         right_eye = landmarks[RIGHT_EYE]

#         left_ratio = eye_ratio(left_eye)
#         right_ratio = eye_ratio(right_eye)
#         avg_ratio = (left_ratio + right_ratio) / 2

#         # Sleep detection
#         if avg_ratio < 0.25:
#             sleep_counter += 1
#             if sleep_counter > 15:
#                 status = "Sleepy 😴"
#         else:
#             sleep_counter = 0

#         # Head direction
#         nose_x = landmarks[1][0]
#         if nose_x < w * 0.4:
#             status = "Looking Left 👀"
#         elif nose_x > w * 0.6:
#             status = "Looking Right 👀"

#     else:
#         if time.time() - inactive_start > 5:
#             status = "Not Present ❌"

#     cv2.putText(frame, status, (30, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Behavior Monitor", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Behavior monitor stopped.")


