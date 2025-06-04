import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

class UprightHandRotationDetector:
    def __init__(self, rotation_threshold=12, smoothing_frames=8):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Rotation detection parameters
        self.rotation_threshold = rotation_threshold  # degrees per frame
        self.palm_angle_history = deque(maxlen=smoothing_frames)
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
        vertical_ratio = dy / (dx + 0.001)  # Avoid division by zero
        
        # Hand is considered upright if fingers are significantly above wrist
        return vertical_ratio > 1.5 and dy > 0.1
    
    def calculate_palm_rotation_angle(self, landmarks):
        """
        Calculate the rotation angle of the palm when hand is upright.
        Uses the line across the palm (from index to pinky base) as reference.
        """
        # Get landmarks for palm width calculation
        wrist = landmarks[0]
        index_mcp = landmarks[5]   # Index finger base
        pinky_mcp = landmarks[17]  # Pinky finger base
        
        # Alternative: use thumb and pinky tips for more pronounced rotation
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        
        # Use the line between thumb tip and pinky tip for rotation
        dx = pinky_tip.x - thumb_tip.x
        dy = pinky_tip.y - thumb_tip.y
        
        # Calculate angle of this line (palm rotation)
        angle = math.degrees(math.atan2(dy, dx))
        
        return angle
    
    def detect_rotation(self, current_angle):
        """Detect if palm is rotating based on angle changes."""
        self.palm_angle_history.append(current_angle)
        
        if len(self.palm_angle_history) < 3:
            return False
        
        # Calculate angular velocity (change in angle over time)
        recent_angles = list(self.palm_angle_history)[-3:]
        
        # Handle angle wraparound (e.g., -179 to 179 degrees)
        angle_diffs = []
        for i in range(1, len(recent_angles)):
            diff = recent_angles[i] - recent_angles[i-1]
            
            # Normalize angle difference to [-180, 180]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
                
            angle_diffs.append(abs(diff))
        
        # Check if average angular velocity exceeds threshold
        avg_angular_velocity = sum(angle_diffs) / len(angle_diffs)
        
        return avg_angular_velocity > self.rotation_threshold
    
    def draw_palm_line(self, frame, landmarks):
        """Draw a line across the palm to visualize rotation."""
        h, w = frame.shape[:2]
        
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        
        # Convert normalized coordinates to pixel coordinates
        thumb_x = int(thumb_tip.x * w)
        thumb_y = int(thumb_tip.y * h)
        pinky_x = int(pinky_tip.x * w)
        pinky_y = int(pinky_tip.y * h)
        
        # Draw line across palm
        cv2.line(frame, (thumb_x, thumb_y), (pinky_x, pinky_y), (255, 0, 255), 3)
        
        # Draw center point (reference)
        center_x = (thumb_x + pinky_x) // 2
        center_y = (thumb_y + pinky_y) // 2
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)
    
    def run(self):
        """Main loop to capture video and detect hand rotation."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Upright Hand Rotation Detector Started!")
        print("Point your hand upward (fingers up, wrist vertical) and rotate your palm.")
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
            
            # Reset detection flags
            self.rotation_detected = False
            self.hand_upright = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Check if hand is in upright position
                    self.hand_upright = self.is_hand_upright(hand_landmarks.landmark)
                    
                    if self.hand_upright:
                        # Calculate palm rotation angle
                        palm_angle = self.calculate_palm_rotation_angle(hand_landmarks.landmark)
                        
                        # Detect rotation
                        self.rotation_detected = self.detect_rotation(palm_angle)
                        
                        # Draw palm reference line
                        self.draw_palm_line(frame, hand_landmarks.landmark)
                        
                        # Display palm angle
                        cv2.putText(
                            frame, 
                            f"Palm Angle: {palm_angle:.1f}°", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (255, 255, 255), 
                            2
                        )
                        
                        # Display rotation status
                        if self.rotation_detected:
                            cv2.putText(
                                frame, 
                                "PALM ROTATING!", 
                                (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1.0, 
                                (0, 255, 0), 
                                3
                            )
                            # Draw rotating indicator
                            cv2.circle(frame, (100, 120), 25, (0, 255, 0), 4)
                            cv2.putText(frame, "↻", (90, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        else:
                            cv2.putText(
                                frame, 
                                "Palm Static", 
                                (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1.0, 
                                (0, 255, 255), 
                                3
                            )
                        
                        cv2.putText(
                            frame, 
                            "Hand Upright - Ready!", 
                            (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 255, 0), 
                            2
                        )
                    else:
                        cv2.putText(
                            frame, 
                            "Point hand upward (fingers up)", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 165, 255), 
                            2
                        )
                        # Clear angle history when hand is not upright
                        self.palm_angle_history.clear()
            else:
                cv2.putText(
                    frame, 
                    "No hand detected", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
            
            # Display instructions
            cv2.putText(
                frame, 
                "Hold hand upright and rotate palm", 
                (10, frame.shape[0] - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            cv2.putText(
                frame, 
                "Like turning a doorknob vertically", 
                (10, frame.shape[0] - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            cv2.putText(
                frame, 
                "Press 'q' to quit", 
                (10, frame.shape[0] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            
            # Show the frame
            cv2.imshow('Upright Hand Rotation Detector', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Create and run the upright hand rotation detector
    detector = UprightHandRotationDetector(
        rotation_threshold=8,   # Adjust sensitivity (lower = more sensitive)
        smoothing_frames=6     # Adjust smoothing (higher = more stable)
    )
    detector.run()

if __name__ == "__main__":
    main()
