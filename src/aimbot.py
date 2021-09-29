import numpy as np
import time
import win32api
import win32con
import keyboard

class AimBot:
    def __init__(self, height, width):
        self.window_center = np.array([width, height])
        self.mode = 1
        self.smoothness = 2
        self.confidence = 0.4
        self.prediction = True
        self.last_center = None
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 100, 0, 0, 0)
        print("Aimbot Active")
    
    def get_centers(self, coords):
        """
        Returns the centers of objects detected
            coords: list of coordinates returned by model
            frame: image upon which model was run
        """
        width, height = self.window_center * 2
        mask = coords[:, 4] > self.confidence
        rows = coords[mask, :]
        centers = []
        for row in rows:
            x1 = int(row[0] * width)
            y1 = int(row[1] * height)
            x2 = int(row[2] * width)
            y2 = int(row[3] * height)
            center_x = abs(x2 + x1) // 2
            center_y = abs(y2 + y1) // 2
            centers.append([center_x, center_y])
        return np.array(centers)
    
    def least_distance(self, centers):
        """
        Returns the point the least distance from the crosshair
            centers: numpy array of center points
        """
        distances = np.cumsum((centers - self.window_center)**2, axis=1)[:,1] > 8
        if len(distances) == 0:
            return None
        return centers[np.argsort(distances)][0]
    
    def move_mouse(self, center):
        """
        Moves mouse to center point. Changes the position of the crosshair to the passed point
        """
        # height, width, _ = frame.shape
        # height //= 2
        # width //= 2
        # change = center[0] - width, center[1] - height
        # x_step = int(change[0] / self.smoothness)
        # y_step = int(change[1] / self.smoothness)
        x_step, y_step = ((center - self.window_center) / self.smoothness).astype(int)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_step, y_step, 0, 0)
        if self.prediction and self.last_center is not None:
            self.predict_next_pos(center, np.array(x_step, y_step) + self.window_center)
    
    def predict_next_pos(self, center, next_window_center):
        """
        Predicts the position of the next frames and moves mouse accordingly 
            frames: number of frames to predict
        """
        predicted_center = center + ((center - self.last_center)) / 2
        if np.sqrt(np.sum((predicted_center - center) ** 2)) < 10:
            return
        x_step, y_step = (((predicted_center - next_window_center) / self.smoothness) // 2).astype(int)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x_step, y_step, 0, 0)
        
    
    def scan(self, coords, labels):
        """Scan is the main function that performs all the major tasks of the aimbot"""
        # Hotkeys List
        if keyboard.is_pressed("h"): # Track head
            self.change_mode(1)
        if keyboard.is_pressed("b"): # Track body
            self.change_mode(0)
        if keyboard.is_pressed("up"): # increase smoothness 
            self.smoothness += 0.05
        if keyboard.is_pressed("down"): # decrease smoothness 
            self.smoothness -= 0.05
        if keyboard.is_pressed("right"): # increase confidence threshold
            self.confidence += 0.05
        if keyboard.is_pressed("left"): # decrease confidence threshold
            self.confidence -= 0.05
        if keyboard.is_pressed("p"):
            self.prediction = True
        if keyboard.is_pressed("o"):
            self.prediction = False
        
        # Coordinate Processing
        coords = coords[labels==self.mode] # Select coordinates with specific label
        if len(coords) == 0:
            self.last_center = None
            return # skip if no objects found
        centers = self.get_centers(coords)
        if len(centers) == 0:
            self.last_center = None
            return 
        
        # Mouse Movement
        best_center = self.least_distance(centers) # get the object the least distance from the crosshair
        if best_center is None:
            self.last_center = None
            return
        self.move_mouse(best_center) # Move the mouse to that object
        self.last_center = best_center
    
    def change_mode(self, mode):
        self.mode = mode

if __name__ == "__main__":
    time.sleep(2)
    print("movign mouse")
    ab = AimBot(2560, 1440, 125, 250)
    