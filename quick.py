import camera

"""
control camera
r, f to adjust focus
v to get current focus
t, g to adjust exposure
b to get current exposure value
e to focus
q to exit
w, a, s, d to move roi (caps lock for fine control)
i, j, k, l to change roi size (caps lock for fine control)
e to focus using the roi
"""

cam = camera.init_windows()
camera.control(cam)
cam.release()
