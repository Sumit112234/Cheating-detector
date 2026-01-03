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
warnings.filterwarnings('ignore')

class AIInterviewerCheatingDetector:
    def __init__(self):
        print("="*70)
        print("AI INTERVIEWER - ADVANCED CHEATING DETECTION SYSTEM")
        print("="*70)
        print("Initializing with multiple validation layers...")
        
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
        print("\n[1/6] Loading AI Models...")
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
        except:
            print("  Failed to load YOLO model")
            raise
        
        print(f"  ✓ Models loaded successfully")
        print(f"  YOLO can detect {len(self.yolo.names)} object types")
        
        # ---------- LANDMARKS ----------
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [13, 14, 78, 308]
        
        # ---------- DETECTION PARAMETERS (CAREFULLY TUNED) ----------
        print("\n[2/6] Setting up detection parameters...")
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
        
        # ---------- VISUAL SETTINGS ----------
        self.COLORS = {
            'safe': (0, 255, 0),        # Green
            'warning': (0, 255, 255),   # Yellow
            'danger': (0, 165, 255),    # Orange
            'critical': (0, 0, 255),    # Red
            'info': (255, 255, 255),    # White
            'highlight': (255, 0, 0)    # Blue
        }
        
        # FPS tracking
        self.frame_times = deque(maxlen=30)
        self.fps = 0
        
        # ---------- VALIDATION SYSTEM ----------
        print("\n[3/6] Setting up validation system...")
        self.cheating_incidents = []  # Track all cheating incidents with timestamps
        self.false_positive_filters = {
            'eye_blink_buffer': deque(maxlen=5),  # Allow normal blinking
            'natural_movement_timer': time.time(),
            'last_validation_check': time.time()
        }
        
        # ---------- AUDIT LOGGING ----------
        print("\n[4/6] Setting up audit logging...")
        self.audit_log = []
        self.interview_start_time = time.time()
        
        # Create logs directory
        if not os.path.exists('interview_logs'):
            os.makedirs('interview_logs')
        
        # ---------- MULTI-LAYER STATE TRACKING ----------
        print("\n[5/6] Setting up multi-layer tracking...")
        self.reset_state()
        
        print("\n[6/6] Initialization complete!")
        print("\n" + "="*70)
        print("SYSTEM READY - Starting with conservative detection thresholds")
        print("="*70)
    
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
        
        print("[SYSTEM] Monitoring state reset for new interview")
    
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
    
    def analyze_frame(self, frame):
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
            "audit_notes": []
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
    
    def save_audit_log(self):
        """Save audit log to file"""
        if not self.audit_log:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_logs/interview_{timestamp}.json"
        
        log_data = {
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
        
        print(f"\n[SYSTEM] Audit log saved to {filename}")
        return filename
    
    # ---------- VISUAL DISPLAY ----------
    
    def display_results(self, frame, response):
        """Display analysis results with careful warnings"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Calculate FPS
        current_time = time.time()
        self.frame_times.append(current_time)
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        
        # ---------- STATUS BAR ----------
        status_color = self.COLORS['safe']
        status_text = "NORMAL"
        
        confidence = response.get("cheating_confidence", 0)
        if confidence >= self.CONFIDENCE_LEVELS['CRITICAL']:
            status_color = self.COLORS['critical']
            status_text = "CRITICAL - REVIEW REQUIRED"
        elif confidence >= self.CONFIDENCE_LEVELS['HIGH']:
            status_color = self.COLORS['danger']
            status_text = "HIGH RISK"
        elif confidence >= self.CONFIDENCE_LEVELS['MEDIUM']:
            status_color = self.COLORS['warning']
            status_text = "CAUTION"
        
        # Draw status bar
        cv2.rectangle(display, (0, 0), (w, 40), status_color, -1)
        cv2.putText(display, f"STATUS: {status_text}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS and time
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(display, f"{time_str} | FPS: {self.fps:.1f}", (w-200, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ---------- DETECTION PANEL ----------
        y_offset = 60
        
        # Face info
        if not response["face_detected"]:
            cv2.putText(display, "NO FACE DETECTED", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['critical'], 2)
            y_offset += 25
        else:
            face_text = f"Faces: {response['face_count']}"
            face_color = self.COLORS['critical'] if response['multiple_faces'] else self.COLORS['safe']
            cv2.putText(display, face_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
            y_offset += 25
        
        # Eye state
        eye_color = self.COLORS['critical'] if response['eyes_closed'] else self.COLORS['safe']
        eye_text = "EYES: CLOSED" if response['eyes_closed'] else f"EYES: OPEN (EAR: {response.get('eye_aspect_ratio', 0):.2f})"
        if response['sleepy']:
            eye_text += " - DROWSY"
        cv2.putText(display, eye_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        y_offset += 25
        
        # Gaze direction
        gaze_color = self.COLORS['safe'] if response['gaze_direction'] == 'center' else self.COLORS['warning']
        gaze_text = f"GAZE: {response['gaze_direction'].upper()}"
        cv2.putText(display, gaze_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, gaze_color, 2)
        y_offset += 25
        
        # Attention
        attention = response.get('attention_percentage', 0)
        attention_color = self.COLORS['safe']
        if attention < 70:
            attention_color = self.COLORS['warning']
        if attention < 50:
            attention_color = self.COLORS['danger']
        
        cv2.putText(display, f"ATTENTION: {attention}%", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, attention_color, 2)
        y_offset += 25
        
        # ---------- OBJECT DETECTION ----------
        if response["detected_objects"]:
            cv2.putText(display, "DETECTED OBJECTS:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['info'], 2)
            y_offset += 20
            
            for obj in response["detected_objects"][:3]:  # Show first 3
                obj_color = self.COLORS['critical'] if 'phone' in obj['label'].lower() else self.COLORS['warning']
                obj_text = f"  • {obj['label']} ({obj['confidence']:.1%})"
                cv2.putText(display, obj_text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 1)
                y_offset += 18
        
        # ---------- CONFIDENCE METER ----------
        conf_y = h - 150
        cv2.putText(display, "CHEATING CONFIDENCE:", (10, conf_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['info'], 2)
        
        # Draw confidence bar
        bar_width = 300
        bar_height = 20
        bar_x = 10
        bar_y = conf_y + 10
        
        # Background
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Fill based on confidence
        fill_width = int(bar_width * confidence)
        
        # Color gradient
        if confidence < 0.3:
            fill_color = self.COLORS['safe']
        elif confidence < 0.6:
            fill_color = self.COLORS['warning']
        elif confidence < 0.85:
            fill_color = self.COLORS['danger']
        else:
            fill_color = self.COLORS['critical']
        
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     fill_color, -1)
        
        # Confidence text
        conf_text = f"{confidence:.1%}"
        cv2.putText(display, conf_text, (bar_x + bar_width + 10, bar_y + bar_height - 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fill_color, 2)
        
        # ---------- INCIDENT COUNTER ----------
        incidents = len(self.cheating_incidents)
        if incidents > 0:
            inc_y = h - 100
            inc_color = self.COLORS['critical'] if incidents >= 3 else self.COLORS['warning']
            cv2.putText(display, f"VALIDATED INCIDENTS: {incidents}", (10, inc_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, inc_color, 2)
        
        # ---------- WARNINGS ----------
        if response["warnings"]:
            warning_y = h - 50
            for warning in response["warnings"][-2:]:  # Show last 2 warnings
                cv2.putText(display, f"⚠ {warning}", (10, warning_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['critical'], 2)
                warning_y -= 25
        
        # ---------- CRITICAL ALERT ----------
        if response["requires_human_review"]:
            alert_text = "HUMAN REVIEW REQUIRED!"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            
            # Background
            cv2.rectangle(display, (text_x-10, text_y-40), 
                         (text_x + text_size[0] + 10, text_y + 10), 
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(display, alert_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLORS['critical'], 3)
            
            # Blinking border
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(display, (5, 5), (w-5, h-5), self.COLORS['critical'], 3)
        
        return display
    
    # ---------- MAIN LOOP ----------
    
    def run_interview_monitoring(self, camera_index=0):
        """Main monitoring loop for interview"""
        print("\n" + "="*70)
        print("STARTING INTERVIEW MONITORING")
        print("="*70)
        print("System is running with conservative detection thresholds.")
        print("Multiple validations required before flagging cheating.")
        print("\nControls:")
        print("  • Press 'Q' to quit and save audit log")
        print("  • Press 'R' to reset monitoring")
        print("  • Press 'P' to pause/resume")
        print("  • Press 'S' to show system status")
        print("="*70)
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return
        
        # Optimize camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)  # Lower FPS for more processing time
        
        paused = False
        last_status_print = time.time()
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Camera error")
                        time.sleep(0.1)
                        continue
                    
                    # Analyze frame
                    response = self.analyze_frame(frame)
                    
                    # Display results
                    display_frame = self.display_results(frame, response)
                    
                    # Show frame
                    cv2.imshow('AI Interviewer - Cheating Detection', display_frame)
                    
                    # Periodic status updates
                    current_time = time.time()
                    if current_time - last_status_print > 10.0:  # Every 10 seconds
                        confidence = response.get("cheating_confidence", 0)
                        incidents = len(self.cheating_incidents)
                        
                        if confidence > 0.3 or incidents > 0:
                            print(f"\n[STATUS] Confidence: {confidence:.1%} | Incidents: {incidents}")
                            
                            if confidence >= self.CONFIDENCE_LEVELS['HIGH']:
                                print("  ⚠  WARNING: High cheating confidence detected!")
                            
                            if response.get("requires_human_review", False):
                                print("  ⚠  ALERT: Human review required!")
                        
                        last_status_print = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nEnding interview monitoring...")
                    break
                elif key == ord('r'):
                    self.reset_state()
                    print("[SYSTEM] Monitoring reset - starting fresh")
                elif key == ord('p'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"[SYSTEM] Monitoring {status}")
                elif key == ord('s'):
                    self.print_system_status()
        
        except KeyboardInterrupt:
            print("\n\nInterview monitoring interrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save audit log
            log_file = self.save_audit_log()
            
            # Print final report
            self.print_final_report()
            
            if log_file:
                print(f"\nDetailed audit log saved to: {log_file}")
    
    def print_system_status(self):
        """Print current system status"""
        print("\n" + "="*70)
        print("CURRENT SYSTEM STATUS")
        print("="*70)
        print(f"Total frames processed: {self.state['total_frames']}")
        print(f"Attention level: {self.state['attention_frames']/max(1, self.state['total_frames'])*100:.1f}%")
        print(f"Cheating incidents: {len(self.cheating_incidents)}")
        print(f"Phone detections: {self.state['phone_detections']}")
        print(f"Current confidence: {self.calculate_cheating_confidence():.1%}")
        
        # Recent incidents
        if self.cheating_incidents:
            print("\nRecent incidents (last 5):")
            for incident in self.cheating_incidents[-5:]:
                time_ago = time.time() - incident['timestamp']
                print(f"  • {incident['type']} ({time_ago:.0f}s ago)")
        
        print("="*70)
    
    def print_final_report(self):
        """Print final interview report"""
        print("\n" + "="*70)
        print("INTERVIEW MONITORING - FINAL REPORT")
        print("="*70)
        
        duration = time.time() - self.interview_start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        print(f"Interview Duration: {minutes}m {seconds}s")
        print(f"Total Frames Analyzed: {self.state['total_frames']}")
        print(f"Average Attention: {self.state['attention_frames']/max(1, self.state['total_frames'])*100:.1f}%")
        print(f"Validated Cheating Incidents: {len(self.cheating_incidents)}")
        
        # Incident breakdown
        if self.cheating_incidents:
            incident_types = Counter([i['type'] for i in self.cheating_incidents])
            print("\nIncident Breakdown:")
            for inc_type, count in incident_types.items():
                print(f"  • {inc_type}: {count}")
        
        # Final confidence assessment
        final_confidence = self.calculate_cheating_confidence()
        print(f"\nFinal Cheating Confidence: {final_confidence:.1%}")
        
        if final_confidence >= self.CONFIDENCE_LEVELS['CRITICAL']:
            print("⚠  ASSESSMENT: HIGH PROBABILITY OF CHEATING")
            print("   Recommendation: Disqualify candidate")
        elif final_confidence >= self.CONFIDENCE_LEVELS['HIGH']:
            print("⚠  ASSESSMENT: STRONG EVIDENCE OF CHEATING")
            print("   Recommendation: Further investigation required")
        elif final_confidence >= self.CONFIDENCE_LEVELS['MEDIUM']:
            print("⚠  ASSESSMENT: MODERATE SUSPICION")
            print("   Recommendation: Review flagged incidents")
        else:
            print("✓  ASSESSMENT: NO SIGNIFICANT EVIDENCE OF CHEATING")
        
        print("="*70)

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    print("AI Interviewer Cheating Detection System")
    print("This system uses multiple validation layers for accurate detection")
    print("\nNote: This system is designed for careful assessment.")
    print("Multiple incidents and high confidence required before flagging cheating.")
    
    detector = AIInterviewerCheatingDetector()
    detector.run_interview_monitoring()