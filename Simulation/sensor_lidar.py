import numpy as np
import cv2
import sys
sys.path.append("..")
from Simulation.utils import *

class LidarModel:
    def __init__(self,
            sensor_size = 31,
            start_angle = -120.0,
            end_angle = 120.0,
            max_dist = 250.0,
        ):
        self.sensor_size = sensor_size
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.max_dist = max_dist
    
    def measure(self, img_map, pose):
        if len(img_map.shape) > 2:
            img_map = img_map[:,:,0]
        sense_data = []
        inter = (self.end_angle-self.start_angle) / (self.sensor_size-1)
        for i in range(self.sensor_size):
            theta = pose[2] + self.start_angle + i*inter
            sense_data.append(self._ray_cast(img_map, np.array((pose[0], pose[1])), theta))
        return sense_data
    
    def _ray_cast(self, img_map, pose, theta):
        end = np.array((pose[0] + self.max_dist*np.cos(np.deg2rad(theta)), pose[1] + self.max_dist*np.sin(np.deg2rad(theta))))
        x0, y0 = int(pose[0]), int(pose[1])
        x1, y1 = int(end[0]), int(end[1])
        plist = Bresenham(x0, x1, y0, y1)
        i = 0
        dist = self.max_dist
        for p in plist:
            if p[1] >= img_map.shape[0] or p[0] >= img_map.shape[1] or p[1]<0 or p[0]<0:
                continue
            if img_map[p[1], p[0]] < 0.5:
                tmp = np.power(float(p[0]) - pose[0], 2) + np.power(float(p[1]) - pose[1], 2)
                tmp = np.sqrt(tmp)
                if tmp < dist:
                    dist = tmp
        return dist

if __name__ == "__main__":
    img = cv2.flip(cv2.imread("Maps/map1.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    img = img.astype(float)/255.

    lmodel = LidarModel()
    pose = (100,200,0)
    sdata = lmodel.measure(img, pose)
    plist = EndPoint(pose, [61,-120,120], sdata)
    img_ = img.copy()
    for pts in plist:
        cv2.line(
            img_, 
            (int(1*pose[0]), int(1*pose[1])), 
            (int(1*pts[0]), int(1*pts[1])),
            (0.0,1.0,0.0), 1)
    cv2.circle(img_,(pose[0],pose[1]),5,(0.5,0.5,0.5),3)
    img_ = cv2.flip(img_,0)
    cv2.imshow("Lidar Test", img_)
    k = cv2.waitKey(0)