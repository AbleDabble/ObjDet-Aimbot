import cv2
import numpy as np
import mouse
import keyboard

class TriggerBot:
    def __init__(self, \
            scan_color_max=np.array([65, 255, 255]), scan_color_min=np.array([55,230,195]),\
            mode=0):
        self.scan_color_max = scan_color_max
        self.scan_color_min = scan_color_min
        self.mode = mode # Semi-auto = 0 -- auto = 1
    
    def scan(self, img):
        """Clicks when image cross-hair changes to enemy color"""
        # Hotkey list
        if keyboard.is_pressed('3'): # Changes mode to semi-automatic
            self.mode = 0
        if keyboard.is_pressed('4'): # Changes mode to automatic
            self.mode = 1
        
        r, c, _ = img.shape
        img = img[r//2-2:r//2+2, c//2-2:c//2+2, :] # get a 4 x 4 grid of pixels in the center of the image
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.scan_color_min, self.scan_color_max)
        if np.any(mask > 0): # check if any of the pixels are within the color range
            if self.mode == 0:
                mouse.click()
            else:
                mouse.press()
        else:
            mouse.release()
