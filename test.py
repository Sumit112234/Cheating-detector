import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
from collections import deque
import json

class DirectCameraMonitor:
    def __init__(self):
        print("Initializing Camera Monitor...")
        
        # ---------- MODELS ----------
        print("Loading FaceMesh model...")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("Loading YOLO model...")
        self.yolo = YOLO("yolov8n.pt")  # Will download automatically if not present
        print(f"YOLO classes available: {list(self.yolo.names.values())[:10]}...")
        
        # ---------- LANDMARKS ----------
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [13, 14, 78, 308]
        
        # ---------- DETECTION PARAMETERS ----------
        self.EYE_CLOSED_THRESHOLD = 0.22
        self.EYE_CLOSED_TIME_THRESHOLD = 2.0
        self.MOUTH_OPEN_THRESHOLD = 0.65
        self.LOOKING_AWAY_MARGIN = 0.40
        self.FACE_MISSING_TIME_THRESHOLD = 3.0
        
        # ---------- SUSPICIOUS OBJECTS ----------
        self.SUSPICIOUS_OBJECTS = {
            "cell phone", "mobile phone", "phone",
            "laptop", "notebook",
            "book",
            "bottle", "cup", "glass",
            "mouse", "keyboard",
            "remote"
        }
        
        # ---------- STATE ----------
        self.reset_state()
        
        # For FPS calculation
        self.frame_times = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.fps = 0
        
        # Colors for display
        self.COLORS = {
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'white': (255, 255, 255),
            'orange': (0, 165, 255)
        }
    
    def reset_state(self):
        """Reset all monitoring state"""
        self.state = {
            "eyes_closed_since": None,
            "last_face_seen": time.time(),
            "attention_frames": 0,
            "total_frames": 0,
            "phone_hits": 0,
            "suspicious_object_hits": 0,
            "cheat_score": 0
        }
        print("State reset complete.")
    
    def eye_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR)"""
        try:
            eye = np.array(eye_points)
            v1 = np.linalg.norm(eye[1] - eye[5])
            v2 = np.linalg.norm(eye[2] - eye[4])
            h = np.linalg.norm(eye[0] - eye[3])
            if h == 0:
                return 0.3
            return (v1 + v2) / (2.0 * h)
        except:
            return 0.3
    
    def mouth_ratio(self, mouth_points):
        """Calculate Mouth Aspect Ratio (MAR)"""
        try:
            mouth = np.array(mouth_points)
            v = np.linalg.norm(mouth[0] - mouth[1])
            h = np.linalg.norm(mouth[2] - mouth[3])
            if h == 0:
                return 0.3
            return v / h
        except:
            return 0.3
    
    def get_gaze_direction(self, nose_landmark, image_width):
        """Determine gaze direction"""
        nose_x_percentage = nose_landmark[0] / image_width
        if nose_x_percentage < (0.5 - self.LOOKING_AWAY_MARGIN/2):
            return "left"
        elif nose_x_percentage > (0.5 + self.LOOKING_AWAY_MARGIN/2):
            return "right"
        else:
            return "center"
    
    def analyze_frame(self, frame):
        """Analyze a single frame and return results"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize response
        response = {
            "face_detected": True,
            "face_missing": False,
            "multiple_faces": False,
            "face_count": 0,
            "looking_away": False,
            "gaze_direction": "center",
            "attention": 0,
            "sleepy": False,
            "eyes_closed": False,
            "eye_aspect_ratio": 0.0,
            "yawning": False,
            "mouth_open": False,
            "mouth_aspect_ratio": 0.0,
            "phone_detected": False,
            "suspicious_objects_detected": False,
            "detected_objects": [],
            "cheating": False,
            "cheat_score": 0,
            "warnings": []
        }
        
        # Process face mesh
        res = self.face_mesh.process(rgb)
        self.state["total_frames"] += 1
        current_time = time.time()
        
        # ---------- FACE DETECTION ----------
        if not res.multi_face_landmarks:
            response["face_detected"] = False
            response["face_count"] = 0
            if current_time - self.state["last_face_seen"] > self.FACE_MISSING_TIME_THRESHOLD:
                response["face_missing"] = True
                response["warnings"].append("Face missing for too long")
                self.state["cheat_score"] = min(100, self.state["cheat_score"] + 2)
            self.finalize_response(response)
            return response
        
        # Update face seen time
        self.state["last_face_seen"] = current_time
        face_count = len(res.multi_face_landmarks)
        response["face_count"] = face_count
        
        if face_count > 1:
            response["multiple_faces"] = True
            response["warnings"].append(f"Multiple faces detected ({face_count})")
            self.state["cheat_score"] = min(100, self.state["cheat_score"] + 5)
        
        # Analyze primary face
        face = res.multi_face_landmarks[0]
        
        try:
            lm = np.array([[int(l.x * w), int(l.y * h)] for l in face.landmark])
        except:
            self.finalize_response(response)
            return response
        
        # ---------- EYE ANALYSIS ----------
        left_ear = self.eye_ratio(lm[self.LEFT_EYE])
        right_ear = self.eye_ratio(lm[self.RIGHT_EYE])
        avg_ear = (left_ear + right_ear) / 2.0
        response["eye_aspect_ratio"] = round(avg_ear, 3)
        
        # Check for closed eyes
        if avg_ear < self.EYE_CLOSED_THRESHOLD:
            response["eyes_closed"] = True
            if not self.state["eyes_closed_since"]:
                self.state["eyes_closed_since"] = current_time
            elif current_time - self.state["eyes_closed_since"] > self.EYE_CLOSED_TIME_THRESHOLD:
                response["sleepy"] = True
                response["warnings"].append("Drowsiness detected (eyes closed)")
                self.state["cheat_score"] = min(100, self.state["cheat_score"] + 3)
        else:
            self.state["eyes_closed_since"] = None
        
        # ---------- MOUTH ANALYSIS ----------
        mar = self.mouth_ratio(lm[self.MOUTH])
        response["mouth_aspect_ratio"] = round(mar, 3)
        
        if mar > self.MOUTH_OPEN_THRESHOLD:
            response["mouth_open"] = True
            response["yawning"] = True
            response["warnings"].append("Yawning detected")
        
        # ---------- GAZE DIRECTION ----------
        gaze_dir = self.get_gaze_direction(lm[1], w)
        response["gaze_direction"] = gaze_dir
        
        if gaze_dir == "center":
            self.state["attention_frames"] += 1
            response["looking_away"] = False
        else:
            response["looking_away"] = True
            response["warnings"].append(f"Looking {gaze_dir}")
            self.state["cheat_score"] = min(100, self.state["cheat_score"] + 1)
        
        # Calculate attention percentage
        if self.state["total_frames"] > 0:
            response["attention"] = int((self.state["attention_frames"] / self.state["total_frames"]) * 100)
        
        # ---------- OBJECT DETECTION ----------
        try:
            results = self.yolo(frame, conf=0.35, verbose=False)[0]
            detected_items = []
            
            if hasattr(results, 'boxes') and results.boxes is not None:
                for box in results.boxes:
                    cls_idx = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = self.yolo.names[cls_idx]
                    
                    detected_items.append({
                        "label": label,
                        "confidence": round(confidence, 2),
                        "bbox": box.xyxy[0].cpu().numpy().tolist() if hasattr(box, 'xyxy') else []
                    })
                    
                    # Check for suspicious objects (case-insensitive)
                    label_lower = label.lower()
                    for suspicious_item in self.SUSPICIOUS_OBJECTS:
                        if suspicious_item in label_lower:
                            response["suspicious_objects_detected"] = True
                            if any(phone_word in label_lower for phone_word in ["phone", "mobile"]):
                                response["phone_detected"] = True
                                response["warnings"].append(f"Phone detected: {label}")
                                self.state["phone_hits"] += 1
                                self.state["cheat_score"] = min(100, self.state["cheat_score"] + 5)
                            else:
                                response["warnings"].append(f"Suspicious object: {label}")
                                self.state["suspicious_object_hits"] += 1
                                self.state["cheat_score"] = min(100, self.state["cheat_score"] + 3)
                            break
            
            response["detected_objects"] = detected_items
            
        except Exception as e:
            response["warnings"].append(f"YOLO error: {str(e)[:50]}")
        
        self.finalize_response(response)
        return response
    
    def finalize_response(self, response):
        """Finalize the response with cheat score"""
        # Apply decay
        self.state["cheat_score"] = max(0, self.state["cheat_score"] - 0.1)
        self.state["cheat_score"] = min(self.state["cheat_score"], 100)
        response["cheat_score"] = int(self.state["cheat_score"])
        
        # Check cheating conditions
        cheating_conditions = [
            self.state["cheat_score"] >= 15,
            self.state["phone_hits"] >= 3,
            self.state["suspicious_object_hits"] >= 5,
            response.get("multiple_faces", False) and self.state["cheat_score"] > 5
        ]
        
        if any(cheating_conditions):
            response["cheating"] = True
            if "CHEATING DETECTED" not in response["warnings"]:
                response["warnings"].append("CHEATING DETECTED")
        
        response["total_frames"] = self.state["total_frames"]
        response["phone_hits"] = self.state["phone_hits"]
        response["timestamp"] = time.time()
    
    def display_results(self, frame, response):
        """Display analysis results on the frame"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Calculate FPS
        current_time = time.time()
        self.frame_times.append(current_time)
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        
        # ---------- HEADER INFO ----------
        # Timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['green'], 2)
        
        # FPS
        cv2.putText(display, f"FPS: {self.fps:.1f}", (w-120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['green'], 2)
        
        # Frame counter
        cv2.putText(display, f"Frame: {self.state['total_frames']}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['white'], 2)
        
        # ---------- FACE INFO ----------
        y_offset = 90
        if not response["face_detected"]:
            cv2.putText(display, "NO FACE DETECTED", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['red'], 2)
            y_offset += 30
        else:
            face_text = f"Faces: {response['face_count']}"
            color = self.COLORS['red'] if response['face_count'] > 1 else self.COLORS['green']
            cv2.putText(display, face_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # ---------- ATTENTION ----------
        attention = response.get("attention", 0)
        attention_color = self.COLORS['green']
        if attention < 60:
            attention_color = self.COLORS['yellow']
        if attention < 30:
            attention_color = self.COLORS['red']
        
        cv2.putText(display, f"Attention: {attention}%", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, attention_color, 2)
        y_offset += 30
        
        # ---------- EYE STATUS ----------
        if response["eyes_closed"]:
            eye_text = "EYES: CLOSED"
            eye_color = self.COLORS['red']
            if response["sleepy"]:
                eye_text += " (SLEEPY)"
        else:
            ear = response.get("eye_aspect_ratio", 0)
            eye_text = f"EYES: OPEN (EAR: {ear:.2f})"
            eye_color = self.COLORS['green']
        
        cv2.putText(display, eye_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        y_offset += 30
        
        # ---------- GAZE DIRECTION ----------
        gaze = response["gaze_direction"]
        gaze_color = self.COLORS['green'] if gaze == "center" else self.COLORS['orange']
        cv2.putText(display, f"Looking: {gaze.upper()}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
        y_offset += 30
        
        # ---------- MOUTH STATUS ----------
        if response["mouth_open"]:
            mar = response.get("mouth_aspect_ratio", 0)
            mouth_text = f"MOUTH: OPEN (MAR: {mar:.2f})"
            mouth_color = self.COLORS['orange']
            if response["yawning"]:
                mouth_text = "YAWNING DETECTED!"
                mouth_color = self.COLORS['red']
        else:
            mouth_text = "MOUTH: CLOSED"
            mouth_color = self.COLORS['green']
        
        cv2.putText(display, mouth_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
        y_offset += 30
        
        # ---------- CHEAT SCORE ----------
        cheat_score = response["cheat_score"]
        score_color = self.COLORS['green']
        if cheat_score > 5:
            score_color = self.COLORS['yellow']
        if cheat_score >= 15:
            score_color = self.COLORS['red']
        
        cv2.putText(display, f"Cheat Score: {cheat_score}/100", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        y_offset += 30
        
        # ---------- OBJECT DETECTION ----------
        if response["detected_objects"]:
            cv2.putText(display, "Detected Objects:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['white'], 2)
            y_offset += 25
            
            for i, obj in enumerate(response["detected_objects"][:3]):  # Show first 3
                obj_text = f"  • {obj['label']}: {obj['confidence']:.1%}"
                obj_color = self.COLORS['red'] if obj['label'].lower() in ['cell phone', 'mobile phone', 'phone'] else self.COLORS['yellow']
                cv2.putText(display, obj_text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 1)
                y_offset += 20
        
        # ---------- WARNINGS ----------
        if response["warnings"]:
            warning_y = h - 30
            for warning in reversed(response["warnings"][-3:]):  # Show last 3 warnings
                cv2.putText(display, f"⚠ {warning}", (10, warning_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['red'], 2)
                warning_y -= 30
        
        # ---------- CHEATING ALERT ----------
        if response["cheating"]:
            cv2.putText(display, "⚠ CHEATING DETECTED ⚠", (w//2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLORS['red'], 3)
            # Flash red border
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(display, (0, 0), (w-1, h-1), self.COLORS['red'], 10)
        
        return display
    
    def run(self, camera_index=0):
        """Main monitoring loop"""
        print(f"\n{'='*60}")
        print("DIRECT CAMERA MONITOR")
        print("="*60)
        print("Controls:")
        print("  • Press 'Q' to quit")
        print("  • Press 'R' to reset state")
        print("  • Press 'P' to pause/resume")
        print("  • Press 'S' to save current frame")
        print("="*60)
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        paused = False
        frame_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture frame")
                        time.sleep(0.1)
                        continue
                    
                    frame_count += 1
                    
                    # Analyze frame
                    response = self.analyze_frame(frame)
                    
                    # Display results
                    display_frame = self.display_results(frame, response)
                    
                    # Show frame
                    cv2.imshow('Direct Camera Monitor', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('r'):
                    self.reset_state()
                    print("State reset!")
                elif key == ord('p'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"Monitoring {status}")
                elif key == ord('s'):
                    filename = f"frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('d'):
                    # Print debug info
                    print("\n" + "="*60)
                    print("DEBUG INFO")
                    print(json.dumps(response, indent=2))
                    print("="*60)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print summary
            print("\n" + "="*60)
            print("MONITORING SUMMARY")
            print("="*60)
            print(f"Total frames processed: {self.state['total_frames']}")
            print(f"Final cheat score: {self.state['cheat_score']}")
            print(f"Phone detections: {self.state['phone_hits']}")
            print(f"Suspicious object detections: {self.state['suspicious_object_hits']}")
            print(f"Average attention: {self.state['attention_frames']/max(1, self.state['total_frames'])*100:.1f}%")
            print("="*60)

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    monitor = DirectCameraMonitor()
    monitor.run()



# from flask import Flask, request, jsonify
# import cv2
# import mediapipe as mp
# import numpy as np
# import base64
# import time
# from ultralytics import YOLO
# from flask_cors import CORS
# import logging

# app = Flask(__name__)
# CORS(app, origins=["http://localhost:3000"])

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ---------- MODELS ----------
# try:
#     face_mesh = mp.solutions.face_mesh.FaceMesh(
#         max_num_faces=3,  # Increased for better multi-face detection
#         refine_landmarks=True,
#         static_image_mode=False,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )
#     logger.info("FaceMesh model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load FaceMesh: {e}")
#     raise

# try:
#     # Load a more capable YOLO model if needed. 'yolov8n.pt' is the default nano model.
#     # Consider 'yolov8s.pt' or 'yolov8m.pt' for better accuracy at a slight speed cost.
#     yolo = YOLO("yolov8n.pt")
#     logger.info("YOLO model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load YOLO model: {e}")
#     raise

# # ---------- LANDMARKS ----------
# LEFT_EYE = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# MOUTH = [13, 14, 78, 308]

# # ---------- DETECTION PARAMETERS ----------
# EYE_CLOSED_THRESHOLD = 0.22  # Lowered for more sensitive eye closure detection
# EYE_CLOSED_TIME_THRESHOLD = 2.0  # Seconds eyes must be closed to trigger "sleepy"
# MOUTH_OPEN_THRESHOLD = 0.65  # Threshold for yawning
# LOOKING_AWAY_MARGIN = 0.40  # Percentage from center (0.35 = 35% from each side)
# FACE_MISSING_TIME_THRESHOLD = 3.0  # Seconds

# # ---------- SESSION STATE (single user demo) ----------
# state = {
#     "eyes_closed_since": None,
#     "last_face_seen": time.time(),
#     "attention_frames": 0,
#     "total_frames": 0,
#     "phone_hits": 0,
#     "suspicious_object_hits": 0,
#     "cheat_score": 0
# }

# # ---------- SUSPICIOUS OBJECTS LIST (YOLO Class Names) ----------
# SUSPICIOUS_OBJECTS = {
#     "cell phone",
#     "laptop",
#     "book",
#     "bottle",
#     "cup",
#     "mouse",
#     "keyboard",
#     "remote"
# }
# # You can print all available classes with: print(yolo.names)

# # ---------- UTILS ----------
# def eye_ratio(eye_points):
#     """Calculate Eye Aspect Ratio (EAR). A lower value indicates closed eye."""
#     try:
#         # Convert points to NumPy array for vector operations
#         eye = np.array(eye_points)
#         # Vertical distances
#         v1 = np.linalg.norm(eye[1] - eye[5])
#         v2 = np.linalg.norm(eye[2] - eye[4])
#         # Horizontal distance
#         h = np.linalg.norm(eye[0] - eye[3])
#         if h == 0:
#             return 0.3  # Safe default to avoid division by zero
#         ear = (v1 + v2) / (2.0 * h)
#         return ear
#     except Exception as e:
#         logger.warning(f"Error calculating eye ratio: {e}")
#         return 0.3

# def mouth_ratio(mouth_points):
#     """Calculate Mouth Aspect Ratio (MAR). A higher value indicates open mouth."""
#     try:
#         mouth = np.array(mouth_points)
#         v = np.linalg.norm(mouth[0] - mouth[1])  # Vertical distance
#         h = np.linalg.norm(mouth[2] - mouth[3])  # Horizontal distance
#         if h == 0:
#             return 0.3
#         mar = v / h
#         return mar
#     except Exception as e:
#         logger.warning(f"Error calculating mouth ratio: {e}")
#         return 0.3

# def get_gaze_direction(nose_landmark, image_width):
#     """Determine if the person is looking left, right, or center."""
#     nose_x_percentage = nose_landmark[0] / image_width
#     if nose_x_percentage < (0.5 - LOOKING_AWAY_MARGIN/2):
#         return "left"
#     elif nose_x_percentage > (0.5 + LOOKING_AWAY_MARGIN/2):
#         return "right"
#     else:
#         return "center"

# # ---------- ROUTES ----------
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "Enhanced Camera Monitor API is running!"})

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     try:
#         data = request.json
#         if not data or "image" not in data:
#             return jsonify({"error": "No image data provided"}), 400

#         # Decode image
#         img_bytes = base64.b64decode(data["image"])
#         np_img = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#         if img is None:
#             return jsonify({"error": "Invalid image data"}), 400

#         h, w, _ = img.shape
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Initialize response with all possible parameters
#         response = {
#             # Face and Attention
#             "face_detected": True,
#             "face_missing": False,
#             "multiple_faces": False,
#             "face_count": 0,
#             "looking_away": False,
#             "gaze_direction": "center",
#             "attention": 0,
#             # Drowsiness
#             "sleepy": False,
#             "eyes_closed": False,
#             "eye_aspect_ratio": 0.0,
#             # Mouth Activity
#             "yawning": False,
#             "mouth_open": False,
#             "mouth_aspect_ratio": 0.0,
#             # Object Detection
#             "phone_detected": False,
#             "suspicious_objects_detected": False,
#             "detected_objects": [],
#             # Cheating Summary
#             "cheating": False,
#             "cheat_score": 0,
#             "warnings": []
#         }

#         # Process face mesh
#         res = face_mesh.process(rgb)
#         state["total_frames"] += 1
#         current_time = time.time()

#         # ---------- FACE DETECTION ----------
#         if not res.multi_face_landmarks:
#             response["face_detected"] = False
#             response["face_count"] = 0
#             if current_time - state["last_face_seen"] > FACE_MISSING_TIME_THRESHOLD:
#                 response["face_missing"] = True
#                 response["warnings"].append("Face missing for too long")
#                 state["cheat_score"] += 2
#             return finalize(response)

#         # Update face seen time
#         state["last_face_seen"] = current_time
#         face_count = len(res.multi_face_landmarks)
#         response["face_count"] = face_count

#         if face_count > 1:
#             response["multiple_faces"] = True
#             response["warnings"].append(f"Multiple faces detected ({face_count})")
#             state["cheat_score"] += 5

#         # Analyze the primary face (first face detected)
#         face = res.multi_face_landmarks[0]

#         # Convert landmarks to numpy array
#         try:
#             lm = np.array([[int(l.x * w), int(l.y * h)] for l in face.landmark])
#         except Exception as e:
#             logger.warning(f"Error processing landmarks: {e}")
#             return finalize(response)

#         # ---------- EYE ANALYSIS ----------
#         left_ear = eye_ratio(lm[LEFT_EYE])
#         right_ear = eye_ratio(lm[RIGHT_EYE])
#         avg_ear = (left_ear + right_ear) / 2.0
#         response["eye_aspect_ratio"] = round(avg_ear, 3)

#         # Check for closed eyes / sleepiness
#         if avg_ear < EYE_CLOSED_THRESHOLD:
#             response["eyes_closed"] = True
#             if not state["eyes_closed_since"]:
#                 state["eyes_closed_since"] = current_time
#             elif current_time - state["eyes_closed_since"] > EYE_CLOSED_TIME_THRESHOLD:
#                 response["sleepy"] = True
#                 response["warnings"].append("Drowsiness detected (eyes closed)")
#                 state["cheat_score"] += 3
#         else:
#             state["eyes_closed_since"] = None

#         # ---------- MOUTH ANALYSIS ----------
#         mar = mouth_ratio(lm[MOUTH])
#         response["mouth_aspect_ratio"] = round(mar, 3)

#         if mar > MOUTH_OPEN_THRESHOLD:
#             response["mouth_open"] = True
#             response["yawning"] = True
#             response["warnings"].append("Yawning detected")

#         # ---------- GAZE DIRECTION & ATTENTION ----------
#         nose_x = lm[1][0]  # Landmark 1 is the tip of the nose
#         gaze_dir = get_gaze_direction(lm[1], w)
#         response["gaze_direction"] = gaze_dir

#         if gaze_dir == "center":
#             state["attention_frames"] += 1
#             response["looking_away"] = False
#         else:
#             response["looking_away"] = True
#             response["warnings"].append(f"Looking {gaze_dir}")
#             state["cheat_score"] += 1

#         # Calculate attention percentage
#         if state["total_frames"] > 0:
#             response["attention"] = int((state["attention_frames"] / state["total_frames"]) * 100)

#         # ---------- OBJECT DETECTION (YOLO) ----------
#         try:
#             # Run YOLO inference. Adjust 'conf' for sensitivity.
#             results = yolo(img, conf=0.35, verbose=False, classes=None)[0]
#             detected_items = []

#             if hasattr(results, 'boxes') and results.boxes is not None:
#                 for box in results.boxes:
#                     cls_idx = int(box.cls[0])
#                     confidence = float(box.conf[0])
#                     label = yolo.names[cls_idx]

#                     detected_items.append({
#                         "label": label,
#                         "confidence": round(confidence, 2)
#                     })

#                     # Check for suspicious objects
#                     if label in SUSPICIOUS_OBJECTS:
#                         response["suspicious_objects_detected"] = True
#                         if label == "cell phone":
#                             response["phone_detected"] = True
#                             response["warnings"].append("Phone detected")
#                             state["phone_hits"] += 1
#                             state["cheat_score"] += 5
#                         else:
#                             response["warnings"].append(f"Suspicious object: {label}")
#                             state["suspicious_object_hits"] += 1
#                             state["cheat_score"] += 3

#             response["detected_objects"] = detected_items

#         except Exception as e:
#             logger.warning(f"Error in YOLO detection: {e}")

#         return finalize(response)

#     except Exception as e:
#         logger.error(f"Error in analyze endpoint: {e}")
#         return jsonify({"error": "Internal server error", "details": str(e)}), 500

# # ---------- FINALIZE & SCORE MANAGEMENT ----------
# def finalize(response):
#     """Finalize response with cheat score and cheating detection."""
#     # Apply slight decay to cheat score over time
#     state["cheat_score"] = max(0, state["cheat_score"] - 0.05)

#     # Cap the cheat score at a reasonable maximum
#     state["cheat_score"] = min(state["cheat_score"], 100)

#     response["cheat_score"] = int(state["cheat_score"])

#     # Determine if cheating is occurring based on multiple factors
#     cheating_conditions = [
#         state["cheat_score"] >= 15,
#         state["phone_hits"] >= 3,
#         state["suspicious_object_hits"] >= 5,
#         response.get("multiple_faces", False) and state["cheat_score"] > 5
#     ]

#     if any(cheating_conditions):
#         response["cheating"] = True
#         if "CHEATING DETECTED" not in response["warnings"]:
#             response["warnings"].append("CHEATING DETECTED")

#     # Add timestamp and frame info
#     response["timestamp"] = time.time()
#     response["total_frames_processed"] = state["total_frames"]

#     return jsonify(response)

# @app.route("/reset", methods=["POST"])
# def reset_state():
#     """Reset the monitoring state."""
#     global state
#     state = {
#         "eyes_closed_since": None,
#         "last_face_seen": time.time(),
#         "attention_frames": 0,
#         "total_frames": 0,
#         "phone_hits": 0,
#         "suspicious_object_hits": 0,
#         "cheat_score": 0
#     }
#     return jsonify({"message": "State reset successfully", "state": state})

# @app.route("/status", methods=["GET"])
# def get_status():
#     """Get current monitoring status."""
#     return jsonify({
#         "state": state,
#         "thresholds": {
#             "eye_closed": EYE_CLOSED_THRESHOLD,
#             "mouth_open": MOUTH_OPEN_THRESHOLD,
#             "looking_away_margin": LOOKING_AWAY_MARGIN
#         },
#         "server_time": time.time()
#     })

# # ---------- RUN ----------
# if __name__ == "__main__":
#     logger.info("Starting Enhanced Flask Server...")
#     logger.info(f"Monitoring SUSPICIOUS_OBJECTS: {SUSPICIOUS_OBJECTS}")
#     app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)


# import cv2
# import mediapipe as mp
# import pyautogui
# import numpy as np
# import time

# pyautogui.FAILSAFE = False

# # ---------------- CONFIG ----------------
# CAM_W, CAM_H = 640, 480
# SCREEN_W, SCREEN_H = pyautogui.size()
# SMOOTHING = 8
# ZOOM_DELAY = 0.8
# SCROLL_SPEED = 120

# # ---------------- MEDIAPIPE ----------------
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.75,
#     min_tracking_confidence=0.75
# )
# mp_draw = mp.solutions.drawing_utils

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)
# cap.set(3, CAM_W)
# cap.set(4, CAM_H)

# # ---------------- STATE ----------------
# prev_x, prev_y = 0, 0
# last_zoom_time = 0

# # Finger tip landmarks
# TIP_IDS = [4, 8, 12, 16, 20]

# def fingers_up(lm):
#     fingers = []

#     # Thumb
#     fingers.append(1 if lm[TIP_IDS[0]].x > lm[TIP_IDS[0] - 1].x else 0)

#     # Other fingers
#     for i in range(1, 5):
#         fingers.append(1 if lm[TIP_IDS[i]].y < lm[TIP_IDS[i] - 2].y else 0)

#     return fingers  # list of 0/1

# print("STRICT HAND CONTROL v2 STARTED")
# print("1 finger  -> Mouse move")
# print("2 fingers -> Zoom OUT")
# print("3 fingers -> Zoom IN")
# print("5 fingers -> Scroll DOWN")
# print("0 fingers -> Scroll UP")
# print("ESC to exit")

# # ---------------- MAIN LOOP ----------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     if result.multi_hand_landmarks:
#         hand = result.multi_hand_landmarks[0]
#         lm = hand.landmark

#         fingers = fingers_up(lm)
#         total_fingers = sum(fingers)
#         now = time.time()

#         # ---------- 1️⃣ MOUSE MOVE (ONLY INDEX) ----------
#         if fingers == [0, 1, 0, 0, 0]:
#             ix = int(lm[8].x * CAM_W)
#             iy = int(lm[8].y * CAM_H)

#             screen_x = np.interp(ix, (0, CAM_W), (0, SCREEN_W))
#             screen_y = np.interp(iy, (0, CAM_H), (0, SCREEN_H))

#             curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
#             curr_y = prev_y + (screen_y - prev_y) / SMOOTHING
#             pyautogui.moveTo(curr_x, curr_y)
#             prev_x, prev_y = curr_x, curr_y

#             cv2.putText(frame, "MOUSE MOVE", (20, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#         # ---------- 2️⃣ ZOOM OUT (2 FINGERS) ----------
#         elif total_fingers == 2 and now - last_zoom_time > ZOOM_DELAY:
#             pyautogui.hotkey("ctrl", "-")
#             last_zoom_time = now
#             cv2.putText(frame, "ZOOM OUT", (20, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

#         # ---------- 3️⃣ ZOOM IN (3 FINGERS) ----------
#         elif total_fingers == 3 and now - last_zoom_time > ZOOM_DELAY:
#             pyautogui.hotkey("ctrl", "+")
#             last_zoom_time = now
#             cv2.putText(frame, "ZOOM IN", (20, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

#         # ---------- 4️⃣ SCROLL DOWN (OPEN PALM) ----------
#         elif total_fingers == 5:
#             pyautogui.scroll(-SCROLL_SPEED)
#             cv2.putText(frame, "SCROLL DOWN", (20, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#         # ---------- 5️⃣ SCROLL UP (FIST) ----------
#         elif total_fingers == 0:
#             pyautogui.scroll(SCROLL_SPEED)
#             cv2.putText(frame, "SCROLL UP", (20, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#         mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

#     cv2.imshow("Strict Hand Mouse & PDF Control v2", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
