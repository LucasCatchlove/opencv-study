import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

class UprightHandRotationDetector:
    def __init__(self, rotation_threshold=8, history_size=6):
        self.rotation_threshold = rotation_threshold
        self.palm_angle_history = deque(maxlen=history_size)
        self.rotation_detected = False
        self.hand_upright = False
    
    def is_hand_upright(self, landmarks):
        """Check if the hand is in upright position (fingers pointing up)."""
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        
        # Calculate vertical orientation
        dy = wrist.y - middle_tip.y  # Positive if fingers point up
        dx = abs(middle_tip.x - wrist.x)
        
        # Check if hand is roughly vertical (fingers pointing up)
        vertical_ratio = dy / (dx + 0.001)
        return vertical_ratio > 1.5 and dy > 0.1
    
    def calculate_palm_rotation_angle(self, landmarks):
        """Calculate palm rotation using thumb tip to pinky tip line."""
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        
        dx = pinky_tip.x - thumb_tip.x
        dy = pinky_tip.y - thumb_tip.y
        
        angle = math.degrees(math.atan2(dy, dx))
        return angle
    
    def detect_rotation(self, landmarks):
        """Detect palm rotation when hand is upright."""
        self.hand_upright = self.is_hand_upright(landmarks)
        
        if not self.hand_upright:
            self.palm_angle_history.clear()
            return False, 0, False
        
        palm_angle = self.calculate_palm_rotation_angle(landmarks)
        self.palm_angle_history.append(palm_angle)
        
        if len(self.palm_angle_history) < 3:
            return False, palm_angle, True
        
        # Calculate angular velocity
        recent_angles = list(self.palm_angle_history)[-3:]
        angle_diffs = []
        
        for i in range(1, len(recent_angles)):
            diff = recent_angles[i] - recent_angles[i-1]
            
            # Handle angle wraparound
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
                
            angle_diffs.append(abs(diff))
        
        avg_angular_velocity = sum(angle_diffs) / len(angle_diffs)
        self.rotation_detected = avg_angular_velocity > self.rotation_threshold
        
        return self.rotation_detected, palm_angle, True

class HandMovementDetector:
    def __init__(self, movement_threshold=0.02, history_size=6):
        self.movement_threshold = movement_threshold
        self.y_history = deque(maxlen=history_size)
        self.movement_status = "STATIC"
    
    def get_hand_centroid_y(self, landmarks):
        """Calculate the Y-coordinate of hand's center of mass."""
        y_coords = [landmark.y for landmark in landmarks]
        return sum(y_coords) / len(y_coords)
    
    def detect_vertical_movement(self, landmarks):
        """Detect if hand is moving up or down."""
        current_y = self.get_hand_centroid_y(landmarks)
        self.y_history.append(current_y)
        
        if len(self.y_history) < 3:
            return "STATIC", current_y
        
        # Calculate vertical movement over last few frames
        movement = self.y_history[-1] - self.y_history[-3]
        
        if movement < -self.movement_threshold:
            self.movement_status = "UP"
        elif movement > self.movement_threshold:
            self.movement_status = "DOWN"
        else:
            self.movement_status = "STATIC"
        
        return self.movement_status, current_y

class HandFistDetector:
    def __init__(self):
        self.fist_status = "UNKNOWN"
    
    def is_fist_closed(self, landmarks):
        """Detect if hand is closed in a fist by checking fingertip positions."""
        # Get fingertip and knuckle landmarks
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_mcp = [5, 9, 13, 17]   # Index, Middle, Ring, Pinky knuckles
        
        # Check if fingertips are curled down (below their knuckles)
        fingers_curled = 0
        
        for tip_idx, mcp_idx in zip(finger_tips, finger_mcp):
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            
            # If tip is below (higher Y value) than knuckle, finger is curled
            if tip.y > mcp.y:
                fingers_curled += 1
        
        # Check thumb separately (different anatomy)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        
        # For thumb, check if it's close to palm (lower X distance from wrist)
        wrist = landmarks[0]
        thumb_distance = abs(thumb_tip.x - wrist.x)
        
        if thumb_distance < 0.08:  # Thumb tucked in
            fingers_curled += 1
        
        # Fist if most fingers are curled
        return fingers_curled >= 4
    
    def detect_fist_state(self, landmarks):
        """Detect if hand is open or closed fist."""
        is_closed = self.is_fist_closed(landmarks)
        
        if is_closed:
            self.fist_status = "CLOSED"
        else:
            self.fist_status = "OPEN"
        
        return self.fist_status

class MainHandDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize detection modules
        self.rotation_detector = UprightHandRotationDetector()
        self.movement_detector = HandMovementDetector()
        self.fist_detector = HandFistDetector()
    
    def draw_palm_line(self, frame, landmarks):
        """Draw line across palm to visualize rotation."""
        h, w = frame.shape[:2]
        
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        
        thumb_x = int(thumb_tip.x * w)
        thumb_y = int(thumb_tip.y * h)
        pinky_x = int(pinky_tip.x * w)
        pinky_y = int(pinky_tip.y * h)
        
        cv2.line(frame, (thumb_x, thumb_y), (pinky_x, pinky_y), (255, 0, 255), 3)
        
        center_x = (thumb_x + pinky_x) // 2
        center_y = (thumb_y + pinky_y) // 2
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)
    
    def draw_status_overlay(self, frame, rotation_data, movement_data, fist_status):
        """Draw all detection results on the frame."""
        is_rotating, palm_angle, hand_upright = rotation_data
        movement_status, y_pos = movement_data
        
        # Rotation status
        if hand_upright:
            color = (0, 255, 0) if is_rotating else (0, 255, 255)
            status_text = "ROTATING" if is_rotating else "Static"
            cv2.putText(frame, f"Palm: {status_text} ({palm_angle:.1f}°)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # cv2.putText(frame, "Hand Upright ✓", 
            #            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Point hand upward (fingers up)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Movement status
        movement_colors = {"UP": (0, 255, 0), "DOWN": (0, 100, 255), "STATIC": (128, 128, 128)}
        cv2.putText(frame, f"Movement: {movement_status}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, movement_colors[movement_status], 2)
        
        # Fist status
        fist_color = (255, 0, 0) if fist_status == "CLOSED" else (0, 255, 255)
        cv2.putText(frame, f"Hand: {fist_status}", 
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fist_color, 2)
        
        # Visual indicators
        if hand_upright and is_rotating:
            cv2.circle(frame, (50, 130), 20, (0, 255, 0), 3)
            cv2.putText(frame, "↻", (42, 138), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if movement_status == "UP":
            cv2.arrowedLine(frame, (100, 150), (100, 130), (0, 255, 0), 3)
        elif movement_status == "DOWN":
            cv2.arrowedLine(frame, (100, 130), (100, 150), (0, 100, 255), 3)
        
        # if fist_status == "CLOSED":
        #     cv2.putText(frame, "closed fist", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        # else:
        #     cv2.putText(frame, "open fist", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
    
    def run(self):
        """Main detection loop."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Multi-Feature Hand Detector Started!")
        print("- Rotation: Point hand upward and rotate palm")
        print("- Movement: Move your hand up and down")
        print("- Fist: Open and close your hand")
        print("Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get landmarks
                    landmarks = hand_landmarks.landmark
                    
                    # Run all detectors
                    rotation_data = self.rotation_detector.detect_rotation(landmarks)
                    movement_data = self.movement_detector.detect_vertical_movement(landmarks)
                    fist_status = self.fist_detector.detect_fist_state(landmarks)
                    
                    # Draw palm line if hand is upright
                    if rotation_data[2]:  # hand_upright
                        self.draw_palm_line(frame, landmarks)
                    
                    # Draw status overlay
                    self.draw_status_overlay(frame, rotation_data, movement_data, fist_status)
            else:
                cv2.putText(frame, "No hand detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Instructions
            # cv2.putText(frame, "Point up & rotate | Move up/down | Open/close fist", 
            #            (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('Multi-Feature Hand Detector', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = MainHandDetector()
    detector.run()

if __name__ == "__main__":
    main()
