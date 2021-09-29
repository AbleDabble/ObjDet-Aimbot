import cv2
import mss
from PIL import ImageGrab
import torch
import numpy as np
import time
import mouse
import math
import win32api, win32con, win32.win32gui
S_HEIGHT, S_WIDTH = ImageGrab.grab().size
print(S_HEIGHT, S_WIDTH)

WINDOW_SIZE = 250

BBOX = S_HEIGHT // 2 - WINDOW_SIZE, \
    S_WIDTH // 2 - WINDOW_SIZE, \
    S_HEIGHT // 2 + WINDOW_SIZE, \
    S_WIDTH // 2 + WINDOW_SIZE, 1

from ctypes import windll, Structure, c_long, byref

# win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(0), int(0), 0, 0)

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

def get_mouse_position():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return { 'x': pt.x, 'y': pt.y}


def score_frame(frame, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    #frame = [torch.tensor(frame)]
    results = model(frame)
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    coords = results.xyxyn[0][:, :-1].cpu().numpy()
    return labels, coords

def plot_boxes(frame, labels, coords, model):
    n = len(labels)
    rows, cols, _ = frame.shape
    for i in range(n):
        row = coords[i]
        if row[4] > 0.2:
            continue
        x1 = int(row[0] * cols)
        y1 = int(row[1] * rows)
        x2 = int(row[2] * cols)
        y2 = int(row[3] * rows)
        box_color = (0, 0, 255)
        # classes = model.names
        # label_font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.rectangle(frame, \
            (x1, y1), (x2, y2), \
                box_color, 2)
        # frame = cv2.putText(frame, \
        #     classes[labels[i]], \
        #     (x1,y1), \
        #     label_font, 0.9, box_color)
    return frame

model = torch.hub.load('../../gitstuff/yolov5', 'custom', path='../models/best100.pt', source='local', force_reload=True)
classes = model.names
mouse_movement_amount = 10
rows, cols, _ = 500, 500, 4
# model = torch.hub.load('../../gitstuff/yolov5', 'yolov5s', source='local')

def move_mouse(x_amount, y_amount, steps):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x_amount / steps), int(y_amount / steps), 0, 0)

def get_closest_enemy(coords):

    closest_dist = 10000
    closest_enemy_x = 0
    closest_enemy_y = 0

    for i in coords:
        row = i

        if row[4] < 0.4: 
            continue

        x1 = int(row[0] * cols)
        y1 = int(row[1] * rows)
        x2 = int(row[2] * cols)
        y2 = int(row[3] * rows)

        center_enemy = [int(float(x1 + x2) / 2), int(float(y1 + y2) / 2)]

        dist = math.dist([frame_center_h, frame_center_w], center_enemy)
        if dist < closest_dist:
            closest_dist = dist
            closest_enemy_x = center_enemy[0]
            closest_enemy_y = center_enemy[1]

    return closest_enemy_x, closest_enemy_y


while True:
    # while mouse.is_pressed(button='left'):
    start_time = time.time()
    with mss.mss() as sct:
        img = np.array(sct.grab(BBOX))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labels, coords = score_frame(img, model)
    frame = plot_boxes(img, labels, coords, model)
    frame_center_h, frame_center_w, _ = img.shape
    frame_center_h /= 2
    frame_center_w /= 2

    if len(labels) > 0:
        # Get mouse position
        mouse_x = get_mouse_position()['x']
        mouse_y = get_mouse_position()['y']

        coords = coords[labels==1]

        if len(coords) != 0:
            closest_enemy_x, closest_enemy_y = get_closest_enemy(coords)

            difference_x = (closest_enemy_x  - frame_center_h)
            difference_y = (closest_enemy_y  - frame_center_w)
            
            move_mouse(difference_x, difference_y, 3)

    frame = cv2.putText(frame, f"FPS: {1.0 / (time.time() - start_time):2f}", (10, WINDOW_SIZE - 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), thickness=1)
    cv2.imshow("Screen", img)
    cv2.waitKey(1)
    