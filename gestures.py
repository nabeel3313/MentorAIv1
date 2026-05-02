import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# Euclidean distance helper
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Thumbs Up detection
def is_thumbs_up(hand):
    thumb_tip = hand[4]
    thumb_mcp = hand[2]

    # Other fingers: index, middle, ring, pinky
    fingers_tips = [hand[8], hand[12], hand[16], hand[20]]
    fingers_mcp = [hand[5], hand[9], hand[13], hand[17]]

    thumb_up = thumb_tip.y < thumb_mcp.y
    fingers_folded = all(dist(np.array([tip.x, tip.y, tip.z]),
                              np.array([mcp.x, mcp.y, mcp.z])) < 0.05
                         for tip, mcp in zip(fingers_tips, fingers_mcp))

    return thumb_up and fingers_folded

# Hand Raising: wrist (hand[0]) significantly above nose (face[1])
def is_hand_raised(wrist, nose):
    """
    In MediaPipe, Y=0 is the top of the screen, Y=1 is the bottom.
    A raised hand means the wrist Y is LESS than the nose Y by a clear margin.
    """
    return wrist.y < nose.y - 0.15

# Leaning Forward: check if shoulders are high relative to nose
def is_leaning_forward(left_shoulder, right_shoulder, nose):
    """When someone leans into the camera, their shoulders appear higher (lower Y value)."""
    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    return avg_shoulder_y < nose.y + 0.05

# Eye Aspect Ratio (EAR) for blink detection
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(landmarks, eye_indices):
    p = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    A = dist(p[1], p[5])
    B = dist(p[2], p[4])
    C = dist(p[0], p[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Start Holistic model
with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True) as holistic:

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        behavior = []

        if res.face_landmarks and res.pose_landmarks:
            face = res.face_landmarks.landmark

            # Mouth & Chin
            mouth_center = np.array([(face[13].x + face[14].x)/2,
                                     (face[13].y + face[14].y)/2,
                                     face[13].z])
            chin = np.array([face[152].x, face[152].y, face[152].z])

            # Blink detection
            left_ear = eye_aspect_ratio(face, LEFT_EYE)
            right_ear = eye_aspect_ratio(face, RIGHT_EYE)
            if left_ear < 0.2 or right_ear < 0.2:
                behavior.append("Blink")

            # Hands
            hands = []
            if res.left_hand_landmarks:
                hands.append(res.left_hand_landmarks.landmark)
            if res.right_hand_landmarks:
                hands.append(res.right_hand_landmarks.landmark)

            for hand in hands:
                # Fingertips for nail biting
                fingertips = [hand[8], hand[12], hand[16], hand[20]]
                for tip in fingertips:
                    tip_np = np.array([tip.x, tip.y, tip.z])
                    if dist(tip_np, mouth_center) < 0.04:
                        behavior.append("Biting Nails")

                mcp = np.array([hand[5].x, hand[5].y, hand[5].z])
                wrist = np.array([hand[0].x, hand[0].y, hand[0].z])
                if dist(mcp, chin) < 0.05 or dist(wrist, chin) < 0.06:
                    behavior.append("Hand on Jaw")

                # Thumbs Up detection
                if is_thumbs_up(hand):
                    behavior.append("Thumbs Up")

                # Hand Raising detection
                if res.face_landmarks:
                    nose = res.face_landmarks.landmark[1]
                    if is_hand_raised(hand[0], nose):
                        behavior.append("Raising Hand")

        # Leaning Forward detection (uses pose landmarks)
        if res.pose_landmarks and res.face_landmarks:
            nose = res.face_landmarks.landmark[1]
            left_shoulder = res.pose_landmarks.landmark[11]
            right_shoulder = res.pose_landmarks.landmark[12]
            if is_leaning_forward(left_shoulder, right_shoulder, nose):
                behavior.append("Leaning Forward")

        # Draw labels
        y = 40
        for b in set(behavior):
            cv2.putText(img, b, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            y += 40

        cv2.imshow("Behavior Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
