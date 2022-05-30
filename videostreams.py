import cv2
from pathlib import Path
import numpy as np
import os

class cropStream():
    def __init__(self, source: int or str, ):
        self.source = source
        self.cap = None
        self.frame_width, self.frame_height = int(cap.get())

    def assert_input(self):
        if self.source == 0:
            self.cap = cv2.VideoCapture(self.source)
            return self.cap
        else:
            self.source = Path(self.source)
            self.cap = cv2.VideoCapture(self.source)
            return self.cap

    def get_shape(self):
        

      #returns Video_streams ={Whole_video: X
#                               Portion_Stream: x_crop}