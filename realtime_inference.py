import cv2
import numpy as np
import joblib
from collections import Counter, deque
from pathlib import Path

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FEATURE_COLS = []
for i in range(1, 22):
    FEATURE_COLS.extend([f"x{i}", f"y{i}", f"z{i}"])


def preprocess_landmarks(landmarks_pixel: np.ndarray) -> np.ndarray:
    """
    Apply the same preprocessing steps:
    1. Recenter: subtract x1 from all x, y1 from all y (relative to wrist)
    2. Normalize: divide x,y by dist = sqrt(x12^2 + y12^2), replace 0 dist with 1
    3. Output: 63 features in order x1,y1,z1,...,x21,y21,z21
    """

    row = {}
    for i in range(21):
        row[f"x{i+1}"] = landmarks_pixel[i, 0]
        row[f"y{i+1}"] = landmarks_pixel[i, 1]
        row[f"z{i+1}"] = landmarks_pixel[i, 2]

    # Step 1: Recenter (relative to wrist = landmark 0 = x1,y1)
    x1, y1 = row["x1"], row["y1"]
    X_columns = [f"x{i}" for i in range(1, 22)]
    Y_columns = [f"y{i}" for i in range(1, 22)]

    for col in X_columns:
        row[col] = row[col] - x1
    for col in Y_columns:
        row[col] = row[col] - y1

    # Step 2: Normalize by dist = sqrt(x12^2 + y12^2)
    dist = np.sqrt(row["x12"] ** 2 + row["y12"] ** 2)
    if dist == 0:
        dist = 1.0

    for col in X_columns + Y_columns:
        row[col] = row[col] / dist

    # Step 3: Build feature vector in correct order (63 values, no label)
    features = np.array([row[col] for col in FEATURE_COLS], dtype=np.float32)
    return features.reshape(1, -1)


def landmarks_to_pixel(hand_landmarks, frame_width: int, frame_height: int) -> np.ndarray:
    """Convert MediaPipe normalized landmarks to pixel coordinates (matches dataset format)."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        x = lm.x * frame_width
        y = lm.y * frame_height
        z = lm.z 
        landmarks.append([x, y, z])
    return np.array(landmarks, dtype=np.float32)


def main():
    script_dir = Path(__file__).parent
    model_path = script_dir / "svc_poly_c30_model.joblib"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Run the 'Save model for inference' cell in ML_project.ipynb first.")
        return

    model = joblib.load(model_path)

    PRED_WINDOW = 10
    pred_history = deque(maxlen=PRED_WINDOW)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Real-time inference started. Press 'q' to quit.")
    print("Show your hand to the camera for gesture recognition.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label = "No hand"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
            )

            landmarks_pixel = landmarks_to_pixel(hand_landmarks, w, h)
            features = preprocess_landmarks(landmarks_pixel)
            pred = model.predict(features)[0]

            
            pred_history.append(pred)
            counts = Counter(pred_history)
            label = max(counts, key=counts.get)

        cv2.putText(
            frame,
            f"Gesture: {label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
