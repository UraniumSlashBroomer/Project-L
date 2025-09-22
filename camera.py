import cv2
import mediapipe as mp
import time
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

SAVE_DIR = 'dataset'
os.makedirs(SAVE_DIR, exist_ok=True)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

T = 30
N = 21 # кол-во вершин у одной руки
buffer = []
with open('nums.txt', 'r') as file:
    sample_true_id, sample_false_id = list(map(int, file.readline().split()))

label = 1
recording = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if recording:
                buffer.append([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    elif recording:
        buffer.append([(0.0, 0.0, 0.0)] * N)

    if len(buffer) == T:
        if label == 1:
            np.save(os.path.join(f'{SAVE_DIR}/class_1/class{1}_sample_{str(sample_true_id)}.npy'),
                    np.array(buffer, dtype=np.float32))
            sample_true_id += 1
        else:
            np.save(os.path.join(f'{SAVE_DIR}/class_0/class_{0}_sample_{str(sample_false_id)}.npy'),
                    np.array(buffer, dtype=np.float32))
            sample_false_id += 1

        recording = False
        buffer = []

        print(f'Запись прекратилась')

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        label = 1
        # for i in range(2, 0, -1):
        #     print(f'Запись через: {i}')
        #     time.sleep(0.9)

        print(f'Запись пошла')
        recording = True

    elif key == ord('e'):
        label = 0
        # for i in range(2, 0, -1):
        #    print(f'Запись через: {i}')
        #    time.sleep(0.9)

        print(f'Запись пошла')
        recording = True

    elif key == 27:
        with open('nums.txt', 'w') as file:
            file.writelines(f'{sample_true_id} {sample_false_id}')
        break

cap.release()
cv2.destroyAllWindows()
