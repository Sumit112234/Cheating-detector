import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import screen_brightness_control as sbc

# ---------------- CONFIG ----------------
ACTION_DELAY = 0.8  # seconds between actions
CAMERA_INDEX = 0

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(CAMERA_INDEX)

# Finger tip landmark IDs
TIP_IDS = [4, 8, 12, 16, 20]

last_action_time = 0

def count_fingers(hand_landmarks):
    fingers = []

    # Thumb (x-axis check)
    if hand_landmarks.landmark[TIP_IDS[0]].x > hand_landmarks.landmark[TIP_IDS[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (y-axis check)
    for i in range(1, 5):
        if hand_landmarks.landmark[TIP_IDS[i]].y < hand_landmarks.landmark[TIP_IDS[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

print("Gesture control started")
print("1 finger  -> Volume DOWN")
print("2 fingers -> Volume UP")
print("3 fingers -> Brightness UP")
print("4 fingers -> Brightness DOWN")
print("Press ESC to exit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        finger_count = count_fingers(hand)
        current_time = time.time()

        if current_time - last_action_time > ACTION_DELAY:

            # 🔉 Volume DOWN
            if finger_count == 1:
                pyautogui.press("volumedown")
                print("Volume DOWN")

            # 🔊 Volume UP
            elif finger_count == 2:
                pyautogui.press("volumeup")
                print("Volume UP")

            # ☀️ Brightness UP
            elif finger_count == 3:
                try:
                    current = sbc.get_brightness()[0]
                    sbc.set_brightness(min(current + 10, 100))
                    print("Brightness UP")
                except:
                    print("Brightness control not supported")

            # 🌙 Brightness DOWN
            elif finger_count == 4:
                try:
                    current = sbc.get_brightness()[0]
                    sbc.set_brightness(max(current - 10, 0))
                    print("Brightness DOWN")
                except:
                    print("Brightness control not supported")

            last_action_time = current_time

        cv2.putText(
            frame,
            f"Fingers: {finger_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
