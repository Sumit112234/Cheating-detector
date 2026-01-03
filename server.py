# server.py
import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO
from collections import deque, Counter
import json
from datetime import datetime
import os
import warnings
import base64
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
from contextlib import asynccontextmanager

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class FrameRequest(BaseModel):
    image: str  # base64 encoded image
    session_id: Optional[str] = None
    timestamp: Optional[float] = None

class CheatingResponse(BaseModel):
    cheating_detected: bool
    confidence: float
    warnings: List[str]
    details: Dict[str, Any]
    requires_intervention: bool
    session_id: Optional[str] = None
    timestamp: float

# Lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AI Interviewer Cheating Detection Server...")
    
    # Initialize detector
    global detector
    detector = AIInterviewerCheatingDetector()
    
    # Store active sessions
    app.state.sessions = {}
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Interviewer Cheating Detection Server...")
    if detector:
        detector.save_audit_log()

# Create FastAPI app
app = FastAPI(
    title="AI Interviewer Cheating Detection API",
    description="API for detecting cheating behavior during online interviews",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Add your Next.js origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIInterviewerCheatingDetector:
    def __init__(self):
        logger.info("="*70)
        logger.info("AI INTERVIEWER - ADVANCED CHEATING DETECTION SYSTEM")
        logger.info("="*70)
        logger.info("Initializing with multiple validation layers...")
        
        # Initialize buffers FIRST before reset_state()
        self.eye_state_buffer = deque(maxlen=10)
        self.mouth_state_buffer = deque(maxlen=10)
        self.head_movement_buffer = deque(maxlen=20)
        self.gaze_history = deque(maxlen=50)
        self.object_detection_history = deque(maxlen=30)
        self.behavior_history = deque(maxlen=100)
        
        # ---------- CONFIDENCE LEVELS ----------
        self.CONFIDENCE_LEVELS = {
            'LOW': 0.3,      # Suspicion
            'MEDIUM': 0.6,   # Warning
            'HIGH': 0.85,    # Strong evidence
            'CRITICAL': 0.95 # Near certainty
        }
        
        # ---------- MODELS ----------
        logger.info("\n[1/6] Loading AI Models...")
        # Face detection with higher accuracy
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Load enhanced YOLO model
        try:
            self.yolo = YOLO("yolov8n.pt")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        logger.info(f"  ✓ Models loaded successfully")
        logger.info(f"  YOLO can detect {len(self.yolo.names)} object types")
        
        # ---------- LANDMARKS ----------
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [13, 14, 78, 308]
        
        # ---------- DETECTION PARAMETERS (CAREFULLY TUNED) ----------
        logger.info("\n[2/6] Setting up detection parameters...")
        self.EYE_CLOSED_THRESHOLD = 0.18  # Very conservative for eye closure
        self.EYE_CLOSED_TIME_THRESHOLD = 3.0  # 3 seconds minimum for drowsiness
        
        self.MOUTH_OPEN_THRESHOLD = 0.70  # Higher threshold for yawning
        self.MOUTH_MOVING_THRESHOLD = 0.05  # Threshold for lip movement
        
        self.LOOKING_AWAY_MARGIN = 0.30  # Conservative margin for gaze
        
        self.FACE_MISSING_TIME_THRESHOLD = 4.0  # 4 seconds before flagging
        
        self.EAR_CONSECUTIVE_FRAMES = 10  # Need 10 consecutive frames for confirmation
        self.CONFIRMATION_REQUIRED = 3  # Need 3 separate incidents for cheating flag
        
        # ---------- SUSPICIOUS OBJECTS & ACTIVITIES ----------
        self.SUSPICIOUS_OBJECTS = {
            "cell phone": {"score": 8, "confidence_req": 0.7},
            "mobile phone": {"score": 8, "confidence_req": 0.7},
            "phone": {"score": 8, "confidence_req": 0.7},
            "laptop": {"score": 6, "confidence_req": 0.75},
            "book": {"score": 5, "confidence_req": 0.8},
            "notebook": {"score": 5, "confidence_req": 0.8},
            "paper": {"score": 4, "confidence_req": 0.8},
            "tablet": {"score": 7, "confidence_req": 0.7},
            "headphones": {"score": 3, "confidence_req": 0.8},
            "earphones": {"score": 3, "confidence_req": 0.8},
            "second person": {"score": 10, "confidence_req": 0.9}  # Multiple faces
        }
        
        # FPS tracking
        self.frame_times = deque(maxlen=30)
        self.fps = 0
        
        # ---------- VALIDATION SYSTEM ----------
        logger.info("\n[3/6] Setting up validation system...")
        self.cheating_incidents = []  # Track all cheating incidents with timestamps
        self.false_positive_filters = {
            'eye_blink_buffer': deque(maxlen=5),  # Allow normal blinking
            'natural_movement_timer': time.time(),
            'last_validation_check': time.time()
        }
        
        # ---------- AUDIT LOGGING ----------
        logger.info("\n[4/6] Setting up audit logging...")
        self.audit_log = []
        self.interview_start_time = time.time()
        
        # Create logs directory
        if not os.path.exists('interview_logs'):
            os.makedirs('interview_logs')
        
        # ---------- MULTI-LAYER STATE TRACKING ----------
        logger.info("\n[5/6] Setting up multi-layer tracking...")
        self.reset_state()
        
        logger.info("\n[6/6] Initialization complete!")
        logger.info("\n" + "="*70)
        logger.info("SYSTEM READY - Starting with conservative detection thresholds")
        logger.info("="*70)
    
    def reset_state(self):
        """Reset all monitoring state - call at interview start"""
        self.state = {
            # Face tracking
            "eyes_closed_since": None,
            "last_face_seen": time.time(),
            "face_missing_start": None,
            
            # Attention tracking
            "attention_frames": 0,
            "total_frames": 0,
            "looking_away_frames": 0,
            
            # Object detection
            "phone_detections": 0,
            "suspicious_object_detections": 0,
            "consecutive_suspicious_frames": 0,
            
            # Cheating scoring (conservative)
            "cheating_score": 0,
            "confirmed_cheating_incidents": 0,
            
            # Behavior patterns
            "eye_closed_pattern": [],
            "gaze_deviation_pattern": [],
            "object_detection_pattern": [],
            
            # Timestamps
            "last_object_detection": None,
            "last_gaze_deviation": None,
            "last_mouth_movement": None
        }
        
        # Clear buffers
        self.eye_state_buffer.clear()
        self.mouth_state_buffer.clear()
        self.head_movement_buffer.clear()
        self.gaze_history.clear()
        self.object_detection_history.clear()
        self.behavior_history.clear()
        
        logger.info("[SYSTEM] Monitoring state reset for new interview")
    
    # ---------- CORE DETECTION METHODS ----------
    
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate EAR with validation"""
        try:
            eye = np.array(eye_points)
            if len(eye) < 6:
                return 0.3
            
            # Vertical distances
            v1 = np.linalg.norm(eye[1] - eye[5])
            v2 = np.linalg.norm(eye[2] - eye[4])
            
            # Horizontal distance
            h = np.linalg.norm(eye[0] - eye[3])
            
            if h == 0:
                return 0.3
            
            ear = (v1 + v2) / (2.0 * h)
            
            # Validate EAR range
            if ear < 0 or ear > 0.5:
                return 0.3
                
            return ear
        except:
            return 0.3
    
    def calculate_mouth_aspect_ratio(self, mouth_points):
        """Calculate MAR with validation"""
        try:
            mouth = np.array(mouth_points)
            if len(mouth) < 4:
                return 0.3
            
            v = np.linalg.norm(mouth[0] - mouth[1])
            h = np.linalg.norm(mouth[2] - mouth[3])
            
            if h == 0:
                return 0.3
            
            mar = v / h
            
            # Validate MAR range
            if mar < 0 or mar > 2.0:
                return 0.3
                
            return mar
        except:
            return 0.3
    
    def analyze_gaze_pattern(self, nose_landmark, image_width):
        """Analyze gaze with pattern recognition"""
        if nose_landmark is None:
            return "center"
        
        nose_x_percentage = nose_landmark[0] / image_width
        
        # Conservative bounds
        left_bound = 0.5 - (self.LOOKING_AWAY_MARGIN / 2)
        right_bound = 0.5 + (self.LOOKING_AWAY_MARGIN / 2)
        
        if nose_x_percentage < left_bound:
            direction = "left"
        elif nose_x_percentage > right_bound:
            direction = "right"
        else:
            direction = "center"
        
        # Store in history for pattern analysis
        self.gaze_history.append(direction)
        
        # Check for sustained gaze deviation
        if len(self.gaze_history) >= 10:
            recent_gaze = list(self.gaze_history)[-10:]
            gaze_counts = Counter(recent_gaze)
            
            # If looking away for 8 out of last 10 frames
            if gaze_counts.get("left", 0) >= 8 or gaze_counts.get("right", 0) >= 8:
                return "sustained_" + ("left" if gaze_counts.get("left", 0) >= 8 else "right")
        
        return direction
    
    # ---------- ADVANCED OBJECT DETECTION ----------
    
    def validate_object_detection(self, detections, frame_width, frame_height):
        """Validate object detections to avoid false positives"""
        validated_detections = []
        
        for detection in detections:
            label = detection['label'].lower()
            confidence = detection['confidence']
            
            # Check if object is in suspicious list
            for obj_name, obj_info in self.SUSPICIOUS_OBJECTS.items():
                if obj_name in label and confidence >= obj_info['confidence_req']:
                    # Additional validation for phones
                    if 'phone' in obj_name:
                        # Check size - phones should be relatively small
                        bbox = detection.get('bbox', [])
                        if bbox and len(bbox) >= 4:
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]
                            area = width * height
                            
                            # If phone appears too large, might be false positive
                            if area > (frame_width * frame_height * 0.3):  # >30% of frame
                                continue
                    
                    # Check if this is a new detection or continuation
                    current_time = time.time()
                    if self.state['last_object_detection']:
                        time_diff = current_time - self.state['last_object_detection']
                        if time_diff < 2.0:  # Same object within 2 seconds
                            continue
                    
                    validated_detections.append({
                        'label': label,
                        'confidence': confidence,
                        'object_type': obj_name,
                        'score': obj_info['score']
                    })
                    
                    self.state['last_object_detection'] = current_time
                    break
        
        return validated_detections
    
    # ---------- PATTERN RECOGNITION ----------
    
    def analyze_behavioral_patterns(self):
        """Analyze behavior patterns over time"""
        patterns = {
            'sustained_gaze_deviation': False,
            'frequent_object_detection': False,
            'repeated_eye_closure': False,
            'suspicious_sequence': False
        }
        
        # Check gaze patterns
        if len(self.gaze_history) >= 20:
            gaze_counts = Counter(list(self.gaze_history)[-20:])
            if gaze_counts.get('left', 0) >= 15 or gaze_counts.get('right', 0) >= 15:
                patterns['sustained_gaze_deviation'] = True
        
        # Check object detection frequency
        current_time = time.time()
        recent_detections = [d for d in self.object_detection_history 
                           if current_time - d['timestamp'] < 30]  # Last 30 seconds
        
        if len(recent_detections) >= 3:
            patterns['frequent_object_detection'] = True
        
        # Check eye closure pattern
        if len(self.eye_state_buffer) == self.EAR_CONSECUTIVE_FRAMES:
            closed_frames = sum(1 for state in self.eye_state_buffer if state == 'closed')
            if closed_frames >= 8:  # 80% closed in recent frames
                patterns['repeated_eye_closure'] = True
        
        return patterns
    
    # ---------- CHEATING VALIDATION ----------
    
    def validate_cheating_incident(self, incident_type, evidence):
        """Validate cheating incident before counting it"""
        current_time = time.time()
        
        # Check if this is a duplicate incident
        for incident in self.cheating_incidents[-5:]:  # Check last 5 incidents
            if (incident['type'] == incident_type and 
                current_time - incident['timestamp'] < 5.0):
                return False  # Too recent, likely same incident
        
        # Type-specific validation
        if incident_type == 'eye_closure':
            # Need multiple confirmations for eye closure
            if len([i for i in self.cheating_incidents[-10:] 
                   if i['type'] == 'eye_closure']) >= 2:
                return True
            return False
        
        elif incident_type == 'phone_detected':
            # Phone needs high confidence
            if evidence.get('confidence', 0) >= self.CONFIDENCE_LEVELS['HIGH']:
                return True
        
        elif incident_type == 'multiple_faces':
            # Multiple faces needs very high confidence
            if evidence.get('confidence', 0) >= self.CONFIDENCE_LEVELS['CRITICAL']:
                return True
        
        elif incident_type == 'sustained_gaze_deviation':
            # Gaze deviation needs pattern confirmation
            if evidence.get('duration', 0) > 5.0:  # 5+ seconds
                return True
        
        # Default: medium confidence required
        return evidence.get('confidence', 0) >= self.CONFIDENCE_LEVELS['MEDIUM']
    
    def calculate_cheating_confidence(self):
        """Calculate overall cheating confidence score"""
        if not self.cheating_incidents:
            return 0.0
        
        recent_incidents = [i for i in self.cheating_incidents 
                          if time.time() - i['timestamp'] < 300]  # Last 5 minutes
        
        if not recent_incidents:
            return 0.0
        
        # Weight incidents by type and recency
        total_score = 0
        max_score = 0
        
        for incident in recent_incidents:
            # Recency weight (more recent = higher weight)
            recency = 1.0 - min((time.time() - incident['timestamp']) / 300, 1.0)
            
            # Type weight
            type_weights = {
                'phone_detected': 1.0,
                'multiple_faces': 1.0,
                'sustained_gaze_deviation': 0.8,
                'eye_closure': 0.6,
                'suspicious_object': 0.7,
                'body_movement': 0.5
            }
            
            weight = type_weights.get(incident['type'], 0.5)
            confidence = incident.get('confidence', 0.5)
            
            total_score += recency * weight * confidence
            max_score += weight
        
        if max_score == 0:
            return 0.0
        
        return total_score / max_score
    
    # ---------- MAIN ANALYSIS ----------
    
    def analyze_frame(self, frame, session_id=None):
        """Comprehensive frame analysis with multiple validation layers"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize response with all safety checks
        response = {
            # Detection results
            "face_detected": False,
            "face_count": 0,
            "multiple_faces": False,
            "eyes_closed": False,
            "sleepy": False,
            "looking_away": False,
            "gaze_direction": "center",
            "yawning": False,
            "phone_detected": False,
            "suspicious_objects_detected": False,
            
            # Confidence metrics
            "cheating_confidence": 0.0,
            "validation_level": "none",
            "requires_human_review": False,
            
            # Detailed info
            "eye_aspect_ratio": 0.0,
            "mouth_aspect_ratio": 0.0,
            "detected_objects": [],
            "behavioral_patterns": {},
            "warnings": [],
            "audit_notes": [],
            
            # Session info
            "session_id": session_id,
            "timestamp": time.time(),
            "frame_number": self.state["total_frames"]
        }
        
        self.state["total_frames"] += 1
        current_time = time.time()
        
        # ---------- FACE DETECTION (Layer 1) ----------
        face_results = self.face_mesh.process(rgb)
        
        if not face_results.multi_face_landmarks:
            response["face_detected"] = False
            
            if not self.state["face_missing_start"]:
                self.state["face_missing_start"] = current_time
            elif current_time - self.state["face_missing_start"] > self.FACE_MISSING_TIME_THRESHOLD:
                response["warnings"].append("Face not detected for extended period")
                self.log_audit_event("face_missing", 
                                   f"Duration: {current_time - self.state['face_missing_start']:.1f}s")
            return response
        
        # Reset face missing timer
        self.state["face_missing_start"] = None
        self.state["last_face_seen"] = current_time
        
        # Count faces
        face_count = len(face_results.multi_face_landmarks)
        response["face_count"] = face_count
        response["face_detected"] = True
        
        if face_count > 1:
            response["multiple_faces"] = True
            if face_count == 2:
                self.record_cheating_incident('multiple_faces', {
                    'confidence': 0.9,
                    'count': face_count
                })
                response["requires_human_review"] = True
        
        # Analyze primary face
        primary_face = face_results.multi_face_landmarks[0]
        
        try:
            landmarks = np.array([[int(l.x * w), int(l.y * h)] for l in primary_face.landmark])
        except:
            return response
        
        # ---------- EYE ANALYSIS (Layer 2) ----------
        left_ear = self.calculate_eye_aspect_ratio(landmarks[self.LEFT_EYE])
        right_ear = self.calculate_eye_aspect_ratio(landmarks[self.RIGHT_EYE])
        avg_ear = (left_ear + right_ear) / 2.0
        response["eye_aspect_ratio"] = avg_ear
        
        # Update eye state buffer
        eye_state = 'closed' if avg_ear < self.EYE_CLOSED_THRESHOLD else 'open'
        self.eye_state_buffer.append(eye_state)
        
        if avg_ear < self.EYE_CLOSED_THRESHOLD:
            response["eyes_closed"] = True
            
            if not self.state["eyes_closed_since"]:
                self.state["eyes_closed_since"] = current_time
            elif current_time - self.state["eyes_closed_since"] > self.EYE_CLOSED_TIME_THRESHOLD:
                response["sleepy"] = True
                
                # Validate before recording incident
                if self.validate_cheating_incident('eye_closure', {'confidence': 0.7}):
                    self.record_cheating_incident('eye_closure', {
                        'confidence': 0.7,
                        'duration': current_time - self.state["eyes_closed_since"]
                    })
        else:
            self.state["eyes_closed_since"] = None
        
        # ---------- MOUTH ANALYSIS ----------
        mar = self.calculate_mouth_aspect_ratio(landmarks[self.MOUTH])
        response["mouth_aspect_ratio"] = mar
        
        if mar > self.MOUTH_OPEN_THRESHOLD:
            response["yawning"] = True
        
        # ---------- GAZE ANALYSIS ----------
        gaze_dir = self.analyze_gaze_pattern(landmarks[1], w)
        response["gaze_direction"] = gaze_dir
        
        if gaze_dir != "center":
            response["looking_away"] = True
            self.state["looking_away_frames"] += 1
            
            if 'sustained' in gaze_dir:
                # Check duration
                if not self.state["last_gaze_deviation"]:
                    self.state["last_gaze_deviation"] = current_time
                elif current_time - self.state["last_gaze_deviation"] > 5.0:
                    if self.validate_cheating_incident('suspicious_gaze', {'confidence': 0.8}):
                        self.record_cheating_incident('sustained_gaze_deviation', {
                            'confidence': 0.8,
                            'direction': gaze_dir.replace('sustained_', ''),
                            'duration': current_time - self.state["last_gaze_deviation"]
                        })
        else:
            self.state["looking_away_frames"] = 0
            self.state["last_gaze_deviation"] = None
            self.state["attention_frames"] += 1
        
        # ---------- OBJECT DETECTION (Layer 3) ----------
        try:
            yolo_results = self.yolo(frame, conf=0.4, verbose=False)[0]
            detections = []
            
            if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
                for box in yolo_results.boxes:
                    cls_idx = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = self.yolo.names[cls_idx]
                    
                    detection = {
                        'label': label,
                        'confidence': confidence,
                        'bbox': box.xyxy[0].cpu().numpy().tolist() if hasattr(box, 'xyxy') else []
                    }
                    detections.append(detection)
            
            # Validate detections
            validated = self.validate_object_detection(detections, w, h)
            response["detected_objects"] = validated
            
            for obj in validated:
                if 'phone' in obj['object_type']:
                    response["phone_detected"] = True
                    
                    if self.validate_cheating_incident('phone_detected', obj):
                        self.record_cheating_incident('phone_detected', obj)
                        response["requires_human_review"] = True
                
                response["suspicious_objects_detected"] = True
                
                if obj['object_type'] != 'phone':
                    if self.validate_cheating_incident('suspicious_object', obj):
                        self.record_cheating_incident('suspicious_object', obj)
            
            # Store in history
            if validated:
                self.object_detection_history.append({
                    'timestamp': current_time,
                    'objects': validated
                })
                
        except Exception as e:
            response["audit_notes"].append(f"Object detection error: {str(e)[:50]}")
        
        # ---------- BEHAVIORAL PATTERN ANALYSIS ----------
        response["behavioral_patterns"] = self.analyze_behavioral_patterns()
        
        # Check for suspicious patterns
        patterns = response["behavioral_patterns"]
        if patterns['sustained_gaze_deviation'] and patterns['frequent_object_detection']:
            response["requires_human_review"] = True
            self.record_cheating_incident('suspicious_sequence', {
                'confidence': 0.75,
                'patterns': dict(patterns)
            })
        
        # ---------- FINAL CONFIDENCE CALCULATION ----------
        response["cheating_confidence"] = self.calculate_cheating_confidence()
        
        # Set validation level
        confidence = response["cheating_confidence"]
        if confidence >= self.CONFIDENCE_LEVELS['CRITICAL']:
            response["validation_level"] = "critical"
        elif confidence >= self.CONFIDENCE_LEVELS['HIGH']:
            response["validation_level"] = "high"
        elif confidence >= self.CONFIDENCE_LEVELS['MEDIUM']:
            response["validation_level"] = "medium"
        elif confidence >= self.CONFIDENCE_LEVELS['LOW']:
            response["validation_level"] = "low"
        
        # Attention calculation
        if self.state["total_frames"] > 0:
            attention_pct = (self.state["attention_frames"] / self.state["total_frames"]) * 100
            response["attention_percentage"] = round(attention_pct, 1)
        
        return response
    
    def record_cheating_incident(self, incident_type, evidence):
        """Record a validated cheating incident"""
        incident = {
            'timestamp': time.time(),
            'type': incident_type,
            'evidence': evidence,
            'frame_number': self.state["total_frames"],
            'cheating_score': evidence.get('score', 1) * evidence.get('confidence', 0.5)
        }
        
        self.cheating_incidents.append(incident)
        self.state["confirmed_cheating_incidents"] += 1
        
        # Log to audit trail
        self.log_audit_event("cheating_incident", 
                           f"{incident_type} - Confidence: {evidence.get('confidence', 0):.2f}")
    
    def log_audit_event(self, event_type, details):
        """Log event to audit trail"""
        log_entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details,
            'frame': self.state["total_frames"],
            'state_snapshot': {
                'cheating_score': self.state["cheating_score"],
                'phone_detections': self.state["phone_detections"],
                'attention_frames': self.state["attention_frames"]
            }
        }
        
        self.audit_log.append(log_entry)
    
    def save_audit_log(self, session_id=None):
        """Save audit log to file"""
        if not self.audit_log:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_logs/interview_{session_id or 'default'}_{timestamp}.json"
        
        log_data = {
            'session_id': session_id,
            'interview_start': self.interview_start_time,
            'interview_end': time.time(),
            'duration': time.time() - self.interview_start_time,
            'total_frames': self.state["total_frames"],
            'cheating_incidents': len(self.cheating_incidents),
            'final_confidence': self.calculate_cheating_confidence(),
            'log_entries': self.audit_log,
            'incidents_detail': self.cheating_incidents
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"[SYSTEM] Audit log saved to {filename}")
        return filename

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Interviewer Cheating Detection API",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=CheatingResponse)
async def analyze_frame(request: FrameRequest):
    """
    Analyze a single frame for cheating behavior
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Analyze frame
        analysis_result = detector.analyze_frame(frame, request.session_id)
        
        # Prepare response
        confidence = analysis_result.get("cheating_confidence", 0)
        cheating_detected = confidence >= detector.CONFIDENCE_LEVELS['MEDIUM']
        requires_intervention = analysis_result.get("requires_human_review", False)
        
        # Check if warnings should be included
        warnings = []
        if analysis_result.get("multiple_faces"):
            warnings.append("Multiple faces detected")
        if analysis_result.get("phone_detected"):
            warnings.append("Phone detected")
        if analysis_result.get("eyes_closed") and analysis_result.get("sleepy"):
            warnings.append("Drowsiness detected")
        if not analysis_result.get("face_detected"):
            warnings.append("Face not detected")
        
        return CheatingResponse(
            cheating_detected=cheating_detected,
            confidence=confidence,
            warnings=warnings,
            details=analysis_result,
            requires_intervention=requires_intervention,
            session_id=request.session_id,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/start_session")
async def start_session(session_id: str):
    """
    Start a new interview session
    """
    if session_id in app.state.sessions:
        return {"message": "Session already exists", "session_id": session_id}
    
    # Create new detector instance for this session
    detector.reset_state()
    app.state.sessions[session_id] = {
        "start_time": time.time(),
        "cheating_incidents": 0,
        "last_frame": None
    }
    
    logger.info(f"Started new session: {session_id}")
    return {"message": "Session started", "session_id": session_id}

@app.post("/end_session")
async def end_session(session_id: str):
    """
    End an interview session and save logs
    """
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save audit log
    log_file = detector.save_audit_log(session_id)
    
    # Clean up session
    del app.state.sessions[session_id]
    
    logger.info(f"Ended session: {session_id}")
    return {
        "message": "Session ended",
        "session_id": session_id,
        "log_file": log_file
    }

@app.get("/session_status/{session_id}")
async def get_session_status(session_id: str):
    """
    Get status of a session
    """
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = app.state.sessions[session_id]
    
    return {
        "session_id": session_id,
        "duration": time.time() - session_data["start_time"],
        "cheating_incidents": detector.state["confirmed_cheating_incidents"],
        "total_frames": detector.state["total_frames"],
        "current_confidence": detector.calculate_cheating_confidence()
    }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )