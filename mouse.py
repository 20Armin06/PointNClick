#کنترل موس با انگشت بهینه شده
import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

cv2.namedWindow("Mouse Control",cv2.WINDOW_NORMAL)
# تنظیمات ماوس
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
MOUSE_SMOOTHING_FACTOR = 0.3
CLICK_COOLDOWN = 0.5

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# متغیرها
mouse_position_history = deque(maxlen=5)
last_click_time = 0

# وب‌کم
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            direction = hand_info.classification[0].label

            fingers = [False] * 5

            if direction == "Right":
                fingers[0] = landmarks[4].x < landmarks[3].x
            else:
                fingers[0] = landmarks[4].x > landmarks[3].x

            for i in range(1, 5):
                fingers[i] = landmarks[(4 * i) + 4].y < landmarks[(4 * i) + 2].y

            # فقط انگشت اشاره بالا باشه
            if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                mouse_x = index_tip.x * SCREEN_WIDTH
                mouse_y = index_tip.y * SCREEN_HEIGHT

                if mouse_position_history:
                    prev_x, prev_y = mouse_position_history[-1]
                    mouse_x = prev_x + MOUSE_SMOOTHING_FACTOR * (mouse_x - prev_x)
                    mouse_y = prev_y + MOUSE_SMOOTHING_FACTOR * (mouse_y - prev_y)

                mouse_position_history.append((mouse_x, mouse_y))
                pyautogui.moveTo(mouse_x, mouse_y)

                # اگر شست هم بالا بود -> کلیک چپ
                if fingers[0]:
                    current_time = time.time()
                    if current_time - last_click_time > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_time = current_time

                # افکت‌های بصری روی تصویر
                h, w, _ = frame.shape
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 255), -1)

    cv2.imshow("Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # کلید Esc برای خروج
        break

cap.release()
cv2.destroyAllWindows()
