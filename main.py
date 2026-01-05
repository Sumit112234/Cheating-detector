import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
from collections import deque, Counter
import json
from datetime import datetime
import os
import base64
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Interviewer Cheating Detection API",
    description="Real-time cheating detection for online interviews",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

frontend_url = os.getenv("FRONTEND_URL")


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class FrameRequest(BaseModel):
    image: str  # base64 encoded image
    reset: Optional[bool] = False  # Optional flag to reset state

class CheatingResponse(BaseModel):
    cheating_detected: bool
    confidence: float
    warnings: List[str]
    details: Dict[str, Any]
    timestamp: float

# Global detector instance
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    global detector
    logger.info("Initializing Cheating Detection System...")
    detector = CheatingDetector()

# Cheating Detector Class
class CheatingDetector:
    def __init__(self):
        logger.info("Initializing Cheating Detection System...")
        
        # Initialize face mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load YOLO
        try:
            self.yolo = YOLO("yolov8n.pt")
            logger.info("✓ YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            raise
        
        # Initialize state
        self.reset_state()
        
        # Thresholds - Tuned for better detection
        self.EYE_CLOSED_THRESHOLD = 0.25  # Increased for better eye detection
        self.LOOKING_AWAY_THRESHOLD = 0.25  # More sensitive gaze detection
        self.CHEATING_THRESHOLD = 0.3  # Lower threshold for cheating detection
        self.SUSTAINED_GAZE_FRAMES = 10  # Number of frames for sustained gaze
        
        # Landmarks
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LIPS = [13, 14, 78, 308]  # For mouth detection
        
        # Object detection classes to track
        self.SUSPICIOUS_OBJECTS = [
            'cell phone', 'mobile phone', 'phone', 'laptop', 'book',
            'mouse', 'keyboard', 'tablet', 'person', 'tv', 'monitor',
            'remote', 'headphones', 'earphones'
        ]
        
        logger.info("Cheating detector initialized successfully")
    
    def reset_state(self):
        self.state = {
            "total_frames": 0,
            "eye_state_buffer": deque(maxlen=15),
            "gaze_history": deque(maxlen=30),
            "attention_frames": 0,
            "looking_away_frames": 0,
            "phone_detections": 0,
            "sustained_looking_away": False,
            "consecutive_eye_closed": 0,
            "last_warning_time": time.time()
        }
    
    def analyze_frame(self, frame):
        """Analyze a single frame"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        response = {
            "face_detected": False,
            "eyes_closed": False,
            "looking_away": False,
            "sustained_looking_away": False,
            "phone_detected": False,
            "multiple_faces": False,
            "suspicious_object_detected": False,
            "cheating_confidence": 0.0,
            "warnings": [],
            "detected_objects": [],
            "eye_aspect_ratio": 0.0,
            "mouth_aspect_ratio": 0.0,
            "gaze_direction": "center",
            "attention_percentage": 100.0,
            "frame_number": self.state["total_frames"],
            "detailed_warnings": []
        }
        
        self.state["total_frames"] += 1
        
        # Face detection
        face_results = self.face_mesh.process(rgb)
        
        if not face_results.multi_face_landmarks:
            response["warnings"].append("No face detected - Candidate may have left")
            response["detailed_warnings"].append({
                "type": "face_missing",
                "severity": "high",
                "message": "Face not detected in frame"
            })
            return response
        
        response["face_detected"] = True
        
        # Check for multiple faces
        face_count = len(face_results.multi_face_landmarks)
        if face_count > 1:
            response["multiple_faces"] = True
            response["warnings"].append(f"⚠️ Multiple people detected ({face_count})")
            response["detailed_warnings"].append({
                "type": "multiple_faces",
                "severity": "critical",
                "count": face_count,
                "message": f"Found {face_count} faces - Possible cheating"
            })
        
        # Analyze primary face
        primary_face = face_results.multi_face_landmarks[0]
        
        try:
            landmarks = np.array([[int(l.x * w), int(l.y * h)] for l in primary_face.landmark])
        except:
            return response
        
        # Eye analysis
        left_ear = self.calculate_ear(landmarks[self.LEFT_EYE])
        right_ear = self.calculate_ear(landmarks[self.RIGHT_EYE])
        avg_ear = (left_ear + right_ear) / 2
        response["eye_aspect_ratio"] = avg_ear
        
        # Track eye state
        is_eyes_closed = avg_ear < self.EYE_CLOSED_THRESHOLD
        self.state["eye_state_buffer"].append(is_eyes_closed)
        
        # Check for sustained eye closure
        if is_eyes_closed:
            self.state["consecutive_eye_closed"] += 1
            if self.state["consecutive_eye_closed"] >= 5:  # 5 consecutive frames
                response["eyes_closed"] = True
                response["warnings"].append("👁️ Eyes closed for extended period")
                response["detailed_warnings"].append({
                    "type": "eyes_closed",
                    "severity": "medium",
                    "duration_frames": self.state["consecutive_eye_closed"],
                    "message": "Candidate appears to have eyes closed"
                })
        else:
            self.state["consecutive_eye_closed"] = 0
        
        # Mouth analysis (for talking detection)
        mouth_mar = self.calculate_mouth_aspect_ratio(landmarks[self.LIPS])
        response["mouth_aspect_ratio"] = mouth_mar
        
        # Gaze analysis
        gaze_dir = self.analyze_gaze(landmarks[1], w)
        response["gaze_direction"] = gaze_dir
        
        # Track gaze history
        self.gaze_history.append(gaze_dir)
        
        # Check for looking away
        if gaze_dir != "center":
            response["looking_away"] = True
            self.state["looking_away_frames"] += 1
            
            # Check for sustained looking away
            recent_gaze = list(self.gaze_history)[-self.SUSTAINED_GAZE_FRAMES:]
            gaze_counts = Counter(recent_gaze)
            
            if gaze_counts.get("left", 0) > gaze_counts.get("center", 0) or \
               gaze_counts.get("right", 0) > gaze_counts.get("center", 0):
                response["sustained_looking_away"] = True
                response["warnings"].append(f"👀 Looking {gaze_dir} for extended time")
                response["detailed_warnings"].append({
                    "type": "sustained_gaze",
                    "severity": "medium",
                    "direction": gaze_dir,
                    "duration_frames": self.state["looking_away_frames"],
                    "message": f"Looking {gaze_dir} consistently"
                })
            else:
                if time.time() - self.state["last_warning_time"] > 2:  # Prevent spam
                    response["warnings"].append(f"👀 Looking {gaze_dir}")
                    self.state["last_warning_time"] = time.time()
        else:
            self.state["looking_away_frames"] = 0
            self.state["attention_frames"] += 1
        
        # Object detection
        try:
            detections = self.detect_objects(frame)
            response["detected_objects"] = detections
            
            suspicious_detected = False
            for obj in detections:
                if obj.get('is_suspicious', False):
                    suspicious_detected = True
                    response["suspicious_object_detected"] = True
                    
                    if 'phone' in obj['label'].lower():
                        response["phone_detected"] = True
                        self.state["phone_detections"] += 1
                        warning_msg = f"📱 {obj['label']} detected (confidence: {obj['confidence']:.1%})"
                        response["warnings"].append(warning_msg)
                        response["detailed_warnings"].append({
                            "type": "phone_detected",
                            "severity": "critical",
                            "object": obj['label'],
                            "confidence": obj['confidence'],
                            "message": warning_msg
                        })
                    else:
                        warning_msg = f"⚠️ {obj['label']} detected (confidence: {obj['confidence']:.1%})"
                        response["warnings"].append(warning_msg)
                        response["detailed_warnings"].append({
                            "type": "suspicious_object",
                            "severity": "high",
                            "object": obj['label'],
                            "confidence": obj['confidence'],
                            "message": warning_msg
                        })
                        
        except Exception as e:
            logger.error(f"Object detection error: {e}")
        
        # Calculate attention percentage
        if self.state["total_frames"] > 0:
            attention_pct = (self.state["attention_frames"] / self.state["total_frames"]) * 100
            response["attention_percentage"] = round(attention_pct, 1)
            
            if attention_pct < 60:
                response["warnings"].append(f"📉 Low attention: {attention_pct:.1f}%")
                response["detailed_warnings"].append({
                    "type": "low_attention",
                    "severity": "low",
                    "percentage": attention_pct,
                    "message": f"Low attention level: {attention_pct:.1f}%"
                })
        
        # Calculate cheating confidence
        confidence = self.calculate_confidence(response)
        response["cheating_confidence"] = confidence
        
        # Add timestamp
        response["timestamp"] = time.time()
        
        return response
    
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio - lower values mean eyes are more closed"""
        try:
            eye = np.array(eye_points)
            if len(eye) < 6:
                return 0.4  # Default open eye
            
            # Vertical distances
            v1 = np.linalg.norm(eye[1] - eye[5])
            v2 = np.linalg.norm(eye[2] - eye[4])
            
            # Horizontal distance
            h = np.linalg.norm(eye[0] - eye[3])
            
            if h == 0:
                return 0.4
            
            ear = (v1 + v2) / (2.0 * h)
            # Clamp between reasonable values
            return max(0.1, min(ear, 0.5))
        except:
            return 0.4
    
    def calculate_mouth_aspect_ratio(self, mouth_points):
        """Calculate Mouth Aspect Ratio - higher values mean mouth is open"""
        try:
            mouth = np.array(mouth_points)
            if len(mouth) < 4:
                return 0.3
            
            # Vertical distance between top and bottom lip
            vertical = np.linalg.norm(mouth[0] - mouth[1])
            
            # Horizontal distance between corners
            horizontal = np.linalg.norm(mouth[2] - mouth[3])
            
            if horizontal == 0:
                return 0.3
            
            mar = vertical / horizontal
            return max(0.1, min(mar, 1.0))
        except:
            return 0.3
    
    def analyze_gaze(self, nose_landmark, image_width):
        """Analyze gaze direction"""
        if nose_landmark is None or len(nose_landmark) < 2:
            return "center"
        
        nose_x = nose_landmark[0] / image_width
        
        # More sensitive gaze detection
        if nose_x < 0.4:  # Left 40% of screen
            return "left"
        elif nose_x > 0.6:  # Right 40% of screen
            return "right"
        return "center"
    
    def detect_objects(self, frame):
        """Detect suspicious objects with improved detection"""
        detections = []
        try:
            # Use higher confidence threshold for better accuracy
            results = self.yolo(frame, conf=0.5, verbose=False)[0]
            
            if hasattr(results, 'boxes') and results.boxes is not None:
                for box in results.boxes:
                    cls_idx = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = self.yolo.names[cls_idx]
                    
                    # Check if object is suspicious
                    is_suspicious = False
                    for obj in self.SUSPICIOUS_OBJECTS:
                        if obj.lower() in label.lower():
                            is_suspicious = True
                            break
                    
                    if is_suspicious:
                        # Get bounding box coordinates
                        bbox = []
                        if hasattr(box, 'xyxy'):
                            bbox = box.xyxy[0].cpu().numpy().tolist()
                        
                        detections.append({
                            'label': label,
                            'confidence': confidence,
                            'bbox': bbox,
                            'is_suspicious': True
                        })
                        
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
        
        return detections
    
    def calculate_confidence(self, response):
        """Calculate cheating confidence with weighted factors"""
        confidence = 0.0
        
        # Weighted scoring
        if response["multiple_faces"]:
            confidence += 0.4
        if response["phone_detected"]:
            confidence += 0.35
        if response["suspicious_object_detected"]:
            confidence += 0.25
        if response["sustained_looking_away"]:
            confidence += 0.15
        if response["eyes_closed"]:
            confidence += 0.2
        if response["looking_away"]:
            confidence += 0.1
        if response["attention_percentage"] < 50:
            confidence += 0.1
        
        # Normalize to 0-1
        confidence = min(confidence, 1.0)
        
        # Boost confidence if multiple warnings
        warning_count = len(response["warnings"])
        if warning_count >= 3:
            confidence = min(confidence * 1.3, 1.0)
        
        return confidence

@app.get("/")
async def root():
    return {
        "message": "AI Interviewer Cheating Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze",
            "reset": "POST /reset",
            "status": "GET /status"
        }
    }

@app.post("/analyze", response_model=CheatingResponse)
async def analyze_frame(request: FrameRequest):
    """Analyze a frame for cheating behavior"""
    try:
        # Reset state if requested
        if request.reset and detector:
            detector.reset_state()
            logger.info("Detector state reset")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Analyze frame
        analysis_result = detector.analyze_frame(frame)
        
        # Determine if cheating is detected
        cheating_detected = analysis_result["cheating_confidence"] >= detector.CHEATING_THRESHOLD
        
        return CheatingResponse(
            cheating_detected=cheating_detected,
            confidence=analysis_result["cheating_confidence"],
            warnings=analysis_result["warnings"],
            details=analysis_result,
            timestamp=analysis_result.get("timestamp", time.time())
        )
        
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/reset")
async def reset_detector():
    """Reset the detector state"""
    if detector:
        detector.reset_state()
        return {"message": "Detector state reset successfully", "timestamp": time.time()}
    return {"message": "Detector not initialized"}

@app.get("/status")
async def get_status():
    """Get current detector status"""
    if not detector:
        return {"status": "Detector not initialized"}
    
    attention_pct = (detector.state["attention_frames"] / max(1, detector.state["total_frames"])) * 100
    
    return {
        "status": "running",
        "total_frames": detector.state["total_frames"],
        "phone_detections": detector.state["phone_detections"],
        "attention_percentage": round(attention_pct, 1),
        "consecutive_eye_closed": detector.state["consecutive_eye_closed"],
        "looking_away_frames": detector.state["looking_away_frames"]
    }

if __name__ == "__main__":
    print("hello ji Jai Shree Ram")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
