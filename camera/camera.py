import cv2 as cv
import numpy as np
import focus
from scipy.optimize import minimize_scalar
import time
import matplotlib.pyplot as plt


class WebCam(cv.VideoCapture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_resolution(self, width, height):
        self.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    def read_roi(self, p0, p1):
        ret, raw = self.read()
        roi = raw[p0[1]:p1[1], p0[0]:p1[0]]
        return ret, raw, roi


def focus_measure(frame_name, cam, _focus, roi_control):
    f0 = cam.get(cv.CAP_PROP_FOCUS)
    n = np.abs(f0 - _focus) / 5
    n = int(n)
    cam.set(cv.CAP_PROP_FOCUS, _focus)
    # wait for picture to be in focus
    for i in range(n+1):
        ret, raw = cam.read()
        cv.imshow(frame_name, raw)
        cv.waitKey(1)
    ret, raw, roi = cam.read_roi(*roi_control.get_points())
    cv.imshow(frame_name, raw)
    cv.waitKey(1)
    return focus.measure.brenner_x(roi[:, :, 1])


class ROIController:
    def __init__(self, video_height, video_width):
        self.video_width = video_width
        self.video_height = video_height
        self.roi_width = np.round(video_width*0.1)
        self.roi_height = np.round(video_height*0.1)
        self.x = int(np.round(self.video_width/2))
        self.y = int(np.round(self.video_height / 2))
        self.hh_u = int(np.round(self.roi_height/2))
        self.hh_d = self.hh_u
        self.hw_l = int(np.round(self.roi_width/2))
        self.hw_r = self.hw_l
        self.p0, self.p1 = self.get_points()
        self.memo = None

    def get_points(self):
        p0_x = self.x - self.hw_l
        p0_y = self.y - self.hh_u
        p0 = (p0_x, p0_y)
        p1_x = self.x + self.hw_r
        p1_y = self.y + self.hh_d
        p1 = (p1_x, p1_y)
        return p0, p1

    def move_x(self, dx):
        self.memo = self.x
        self.x += int(dx)
        if self.is_error():
            self.x = self.memo

    def move_y(self, dy):
        self.memo = self.y
        self.y += int(dy)
        if self.is_error():
            self.y = self.memo

    def change_h_roi(self, dh):
        self.memo = (self.hh_u, self.hh_d)
        dh = int(dh)
        if dh % 2 == 0:
            dh_u = int(dh / 2)
            dh_d = dh_u
        else:
            dh_u = int(np.floor(dh/2))
            dh_d = dh_u+1
        self.hh_u += dh_u
        self.hh_d += dh_d
        if self.is_error():
            self.hh_u, self.hh_d = self.memo

    def change_w_roi(self, dw):
        self.memo = (self.hw_l, self.hw_r)
        dw = int(dw)
        if dw % 2 == 0:
            dw_l = int(dw / 2)
            dw_r = dw_l
        else:
            dw_l = int(np.floor(dw / 2))
            dw_r = dw_l + 1
        self.hw_l += dw_l
        self.hw_r += dw_r
        if self.is_error():
            self.hw_l, self.hw_r = self.memo

    def is_error(self):
        p0, p1 = self.get_points()
        p0_x, p0_y = p0
        # print(p0_x, p0_y)
        p1_x, p1_y = p1
        # print(p1_x, p1_y)
        result = False
        if p0_x < 0 \
                or p0_y < 0 \
                or p1_x > self.video_width \
                or p1_y > self.video_height \
                or p1_x - p0_x < 0  \
                or p1_y - p0_y < 0:
            result = True
        # print(result)
        return result
