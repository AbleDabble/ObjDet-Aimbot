import mss
import cv2
import os
import time
from PIL import ImageGrab, Image
import numpy as np
import keyboard
S_HEIGHT, S_WIDTH = ImageGrab.grab().size

WINDOW_HEIGHT = 150
WINDOW_WIDTH = 300

BBOX = S_HEIGHT // 2 - WINDOW_HEIGHT, \
    S_WIDTH // 2 - WINDOW_WIDTH, \
    S_HEIGHT // 2 + WINDOW_HEIGHT, \
    S_WIDTH // 2 + WINDOW_WIDTH

SCAN_COLOR_MAX = np.array([65, 255, 255])
SCAN_COLOR_MIN = np.array([55, 230, 195])

img_count = 0

def grab():
    """Returns numpy array of screenshot"""
    with mss.mss() as sct:
        img = np.array(sct.grab(BBOX))
    return img

def scan(img):
    """Returns True if crosshair green else returns false"""
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[height//2-2:height//2+2, width//2-2:width//2+2]
    mask = cv2.inRange(hsv, SCAN_COLOR_MIN, SCAN_COLOR_MAX)
    if np.any(mask > 0):
        return True
    else:
        return False
    
def found_enemy(img):
    """Saves an image every 3 seconds """
    global img_count
    for _ in range(1):
        time.sleep(0.060)
        cv2.imwrite("../imgs/img_" + str(img_count) + ".jpg", grab())
        img_ = cv2.putText(img, "SCREENSHOT SAVED", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
        cv2.imshow("screen", img_)
        cv2.waitKey(1)
        img_count += 1
        time.sleep(3)

def main():
    global img_count
    while True:
        img = grab()
        if scan(img):
            found_enemy(img)
            print(f'Images collected {img_count}', end='\r')
            
        if keyboard.is_pressed("c"):
            cv2.imwrite("../imgs/img_" + str(img_count) + ".jpg", grab())
            img_count += 1
            print(f"Images collected {img_count}", end = '\r')
        cv2.imshow('screen', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    os.system('cls')
    main()