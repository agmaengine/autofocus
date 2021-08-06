from .camera import *
import cv2 as cv
import numpy as np
from copy import copy
from focus import measure, peak
from os.path import expanduser
import datetime
import time


def init_windows():
    # camera backend other than dshow doesn't let user control focus
    cam = WebCam(0, cv.CAP_DSHOW)
    print('set resolution to 1280x720')
    # cam.set_resolution(1280, 720)
    print('resolution was set')
    print('Turn off autofocus')
    cam.set(cv.CAP_PROP_AUTOFOCUS, 0)
    print('Autofocus was off')
    print('start capturing')
    return cam


def control(cam):
    capture_folder = expanduser('~/Pictures')
    ret, raw = cam.read()
    h, w, c = raw.shape
    roi_control = ROIController(h, w)

    while True:
        ret, raw = cam.read()
        frame_name = 'frame'
        p0, p1 = roi_control.get_points()
        shown = copy(raw)
        cv.rectangle(shown, p0, p1, (0, 0, 255), 1)
        cv.imshow(frame_name, shown)
        key = cv.waitKey(1)
        # hook control
        # stop program
        if key == ord('q'):
            break
        # auto focus
        elif key == ord('e'):
            peak.rule_base_search(lambda x: focus_measure(frame_name, cam, x, roi_control))
        # capture
        elif key == ord('\r') or key == ord(' '):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S')
            cv.imwrite(f'{capture_folder}/{timestamp}.png', raw)
            time.sleep(0.2)
        # manual focus control
        elif key == ord('f'):
            f = cam.get(cv.CAP_PROP_FOCUS)
            cam.set(cv.CAP_PROP_FOCUS, int(f + 5))
            print(cam.get(cv.CAP_PROP_FOCUS))
        elif key == ord('r'):
            f = cam.get(cv.CAP_PROP_FOCUS)
            cam.set(cv.CAP_PROP_FOCUS, int(f - 5))
            print(cam.get(cv.CAP_PROP_FOCUS))
        elif key == ord('v'):
            print(cam.get(cv.CAP_PROP_FOCUS))
        # manual exposure control
        elif key == ord('t'):
            e = cam.get(cv.CAP_PROP_EXPOSURE)
            cam.set(cv.CAP_PROP_EXPOSURE, int(e + 1))
            print(cam.get(cv.CAP_PROP_EXPOSURE))
        elif key == ord('g'):
            e = cam.get(cv.CAP_PROP_EXPOSURE)
            cam.set(cv.CAP_PROP_EXPOSURE, int(e - 1))
            print(cam.get(cv.CAP_PROP_EXPOSURE))
        elif key == ord('b'):
            print(cam.get(cv.CAP_PROP_EXPOSURE))
        # roi control
        elif key == ord('s'):
            roi_control.move_y(10)
        elif key == ord('w'):
            roi_control.move_y(-10)
        elif key == ord('d'):
            roi_control.move_x(10)
        elif key == ord('a'):
            roi_control.move_x(-10)
        elif key == ord('i'):
            roi_control.change_h_roi(10)
        elif key == ord('k'):
            roi_control.change_h_roi(-10)
        elif key == ord('l'):
            roi_control.change_w_roi(10)
        elif key == ord('j'):
            roi_control.change_w_roi(-10)
        # fine roi control (caps lock)
        elif key == ord('S'):
            roi_control.move_y(1)
        elif key == ord('W'):
            roi_control.move_y(-1)
        elif key == ord('D'):
            roi_control.move_x(1)
        elif key == ord('A'):
            roi_control.move_x(-1)
        elif key == ord('I'):
            roi_control.change_h_roi(1)
        elif key == ord('K'):
            roi_control.change_h_roi(-1)
        elif key == ord('L'):
            roi_control.change_w_roi(1)
        elif key == ord('J'):
            roi_control.change_w_roi(-1)

    cv.destroyAllWindows()
