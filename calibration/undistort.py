import cv2
import time
import json
import numpy as np

import os
with open(os.path.join(os.path.dirname(__file__), "fisheye_intrinsics.json")) as _intrinsics:
    intrinsics = json.load(_intrinsics)

camera_mat = np.array(intrinsics["matrix"])
camera_dist = np.array(intrinsics["distortion"])
img_size = np.array(intrinsics["size"])
Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_mat, camera_dist, img_size, np.eye(3), balance=0.0)

def undistort(image):
    return cv2.fisheye.undistortImage(image, camera_mat, camera_dist, Knew=Knew)

if __name__ == "__main__":
    print('connecting to camera')
    cam = cv2.VideoCapture(0)
    print("m")
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("m")

    N = 1
    while True:
        print("m")
        t0 = time.time()
        for i in range(N):
            ret, image = cam.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            out = undistort(image)
        t1 = time.time()
        print("FPS: ", N/(t1-t0), image.shape)
        cv2.imshow('image', out)
        cv2.waitKey(1)

