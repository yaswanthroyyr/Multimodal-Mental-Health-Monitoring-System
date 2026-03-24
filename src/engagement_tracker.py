import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import urllib.request

class EngagementTracker:
    def __init__(self):
        # 1. Download the required Google Model if it doesn't exist
        self.model_path = "face_landmarker.task"
        if not os.path.exists(self.model_path):
            print("⏳ Downloading modern Face Landmarker model... (happens once)")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("✅ Download complete.")
            
        # 2. Initialize the modern MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # 3. MediaPipe Eye Landmark Indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _calculate_ear(self, landmarks, eye_indices, img_w, img_h):
        """Calculates the Eye Aspect Ratio."""
        # Note: The new API uses landmarks[i].x, no need for the .landmark attribute
        pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
        
        vert1 = self._euclidean_distance(pts[1], pts[5])
        vert2 = self._euclidean_distance(pts[2], pts[4])
        horz = self._euclidean_distance(pts[0], pts[3])
        
        if horz == 0: return 0.0
        return (vert1 + vert2) / (2.0 * horz)

    def analyze_frame(self, frame):
        """
        Analyzes a single frame for Head Pose and Blink Rate.
        """
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to the new MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process the frame
        results = self.detector.detect(mp_image)

        telemetry = {
            "face_detected": False,
            "ear": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "status": "No Face",
            "engagement_score": 0.0
        }

        # Check if any faces were found
        if results.face_landmarks: 
            telemetry["face_detected"] = True
            # Get the landmarks of the first detected face
            landmarks = results.face_landmarks[0] 

            # 1. Calculate Eye Aspect Ratio (Blinks/Drowsiness)
            left_ear = self._calculate_ear(landmarks, self.LEFT_EYE, img_w, img_h)
            right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE, img_w, img_h)
            avg_ear = (left_ear + right_ear) / 2.0
            telemetry["ear"] = avg_ear

            # 2. Calculate Head Pose
            nose_tip = landmarks[1]
            chin = landmarks[152]
            left_eye_inner = landmarks[133]
            right_eye_inner = landmarks[362]

            # Yaw (Looking left/right)
            dist_left = abs(nose_tip.x - left_eye_inner.x)
            dist_right = abs(nose_tip.x - right_eye_inner.x)
            yaw_ratio = dist_left / (dist_right + 1e-6)
            telemetry["head_yaw"] = yaw_ratio

            # Pitch (Looking up/down)
            dist_chin = abs(nose_tip.y - chin.y)
            dist_eyes = abs(nose_tip.y - left_eye_inner.y)
            pitch_ratio = dist_chin / (dist_eyes + 1e-6)
            telemetry["head_pitch"] = pitch_ratio

            # 3. Engagement Logic
            score = 100.0
            status = "Focused"

            if avg_ear < 0.22: 
                score -= 40
                status = "Drowsy / Eyes Closed"
            if yaw_ratio < 0.5 or yaw_ratio > 2.0:
                score -= 30
                status = "Looking Away"
            if pitch_ratio < 1.5: 
                score -= 30
                status = "Looking Down"

            telemetry["status"] = status
            telemetry["engagement_score"] = max(0.0, score)

        return telemetry

# Quick test execution
if __name__ == "__main__":
    tracker = EngagementTracker()
    cap = cv2.VideoCapture(0)
    print("🎥 Testing Engagement Tracker. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        data = tracker.analyze_frame(frame)
        
        # Display the results
        cv2.putText(frame, f"Status: {data['status']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Engagement: {data['engagement_score']}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"EAR (Eyes): {data['ear']:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Engagement Tracker Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()