import cv2
import mss
from PIL import ImageGrab
import torch
import numpy as np
import time
from aimbot import AimBot
from triggerbot import TriggerBot

S_HEIGHT, S_WIDTH = ImageGrab.grab().size

WINDOW_HEIGHT = 125
WINDOW_WIDTH = 250

# This is should be fixed to be consistent.
BBOX = S_HEIGHT // 2 - WINDOW_WIDTH, \
    S_WIDTH // 2 - WINDOW_HEIGHT, \
    S_HEIGHT // 2 + WINDOW_WIDTH, \
    S_WIDTH // 2 + WINDOW_HEIGHT

def score_frame(frame, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    results = model(frame)
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    coords = results.xyxyn[0][:, :-1].cpu().numpy()
    return labels, coords

def plot_boxes(frame, labels, coords, model):
    """Plots boxes around detected objects"""
    n = len(labels)
    rows, cols, _ = frame.shape
    for i in range(n):
        row = coords[i]
        if row[4] < 0.2:
            continue
        x1 = int(row[0] * cols)
        y1 = int(row[1] * rows)
        x2 = int(row[2] * cols)
        y2 = int(row[3] * rows)
        box_color = (0, 0, 255)
        frame = cv2.rectangle(frame, \
            (x1, y1), (x2, y2), \
                box_color, 2)
    return frame

def create_display(ab, tb, frame, total_fps, total_frames, fps):
    """Returns image with settings and stats on it"""
    settings = f"Aimbot: {ab.mode} -- Triggerbot: {tb.mode} -- Smoothness: {ab.smoothness:.2f}"
    settings2 = f"Confidence: {ab.confidence:.2f}"
    stats = f"FPS: {fps:.0f} -- Avg FPS: {total_fps / total_frames:.0f}"
    frame = cv2.putText(frame, settings, (10, WINDOW_HEIGHT  * 2- 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0), thickness=1)
    frame = cv2.putText(frame, settings2, (10, WINDOW_HEIGHT * 2 - 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0), thickness=1)
    frame = cv2.putText(frame, stats, (10, WINDOW_HEIGHT * 2 - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (240,222,189), thickness=1)
    return frame

# Initialize objects and models
model = torch.hub.load('../../gitstuff/yolov5', 'custom', path='../models/best200.pt', source='local', force_reload=True).autoshape()

ab = AimBot(WINDOW_HEIGHT, WINDOW_WIDTH)
tb = TriggerBot()

total_frames = 0
total_fps = 0

while True:
    start = time.time()
    with mss.mss() as sct:
        img = np.array(sct.grab(BBOX))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labels, coords = score_frame(img, model)
    tb.scan(img)
    ab.scan(coords, labels)
    frame = plot_boxes(img, labels, coords, model)
    # Calculate framerate
    fps = 1 / (time.time() - start)
    total_fps += fps
    total_frames += 1
    frame = create_display(ab, tb, frame, total_fps, total_frames, fps)
    cv2.imshow("Screen", frame)
    cv2.waitKey(1)
    