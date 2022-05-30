import cv2
from pathlib import Path
import numpy as np
import os


class cropStream:
    def __init__(self, source, cap=None, c_x=0, c_y=0, c_h=100, c_w=100):
        # self.image = None
        self.cap = cap
        self.source = source
        if self.cap is None and self.source == 0:
            self.cap = cv2.VideoCapture(self.source)
        else:
            # self.cap is None and source != 0:
            self.cap = cv2.VideoCapture(Path(source))
        self.camera_open = self.cap.isOpened()
        self.c_h = c_h
        self.c_x = c_x
        self.c_y = c_y
        self.c_w = c_w
        self.width = None
        self.height = None
        self.image = None
        self.crop = None

    # def get_shape(self):
    #     if self.width is None:
    #         self.height, self.width = self.image[:2]
    #     return self.height, self.width

    def get_cropped(self):
        while self.camera_open:
            if self.image and self.crop is None:
                ret, self.image = self.cap.read()

                if ret:
                    self.crop = self.image[self.c_y:self.c_y + self.c_h, self.c_x:self.c_x + self.c_w]

                    cv2.imshow('Original Frame', self.image)
                    cv2.imshow('Cropped Frame', self.crop)
                else:
                    break
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    source = 0
    cropStream(source)


if __name__ == '__main__':
    main()
