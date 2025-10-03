import cv2
import torch
import os
import numpy as np
import mediapipe as mp
from collections import deque
from STGCN import SpatialTemporalGraphConvNetwork

def preprocessing(buffer):
    sample = np.array(buffer, dtype=np.float32)
    # normalization
    wrist = sample[:, 0, :]
    sample = sample - wrist[:, np.newaxis, :]

    ref_point = sample[:, 9, :]
    scale = np.linalg.norm(ref_point - wrist)

    if scale > 0:
        sample = sample / scale

    # [T, N, C] -> [C, T, N]
    sample = np.transpose(sample, (2, 0, 1))
    sample = torch.from_numpy(sample)
    sample = sample[None, :, :, :]

    return sample

def load_model(file_name, threshold, device):
    model = SpatialTemporalGraphConvNetwork(2, 3, 21, threshold, device)
    model_dict = torch.load(file_name, map_location='cpu', weights_only=False)
    model.load_state_dict(model_dict['model_params'])
    
    return model

def inference_model(model, buffer, device):
    model.to(device)
    model.eval()

    sample = preprocessing(buffer)
    sample = sample.to(device)
    
    with torch.no_grad():
        logits = model(sample)
        proba = torch.sigmoid(logits)
        predict = 0
        
        print(proba)
        if proba[0][1] >= model.threshold:
            predict = 1

    return predict


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

N = 21
T = 30
stride = 5
counter = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(file_name='model_for_inference.pth',
                   threshold=0.82,
                   device=device)

buffer = deque()
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
)

print(f'camera fps: {cap.get(cv2.CAP_PROP_FPS)}')

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
        
        buffer.append([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    else:
        buffer.append([(0.0, 0.0, 0.0)] * N)

    if len(buffer) == T + 1:
        buffer.popleft()

    counter += 1
    
    if counter % 5 == 0 and len(buffer) == T:
        counter = 0
        predict = inference_model(model, buffer, device)
    
        print(predict)
   
    key = cv2.waitKey(1)
    cv2.imshow('Frame', frame)
    
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
