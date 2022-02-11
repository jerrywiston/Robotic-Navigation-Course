import numpy as np
import cv2
from Simulator.sensor_lidar import LidarModel
import Mapping.utils
from Mapping.grid_map import GridMap 

if __name__ == "__main__":
    # Read Image
    img = cv2.flip(cv2.imread("Maps/map1.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    img = img.astype(float)/255.

    # Lidar Sensor
    lidar_param = [31, -120.0, 120.0, 250.0]
    lidar = LidarModel(*lidar_param)
    pos = (100,200,0)
    sdata = lidar.measure(img, pos)
    plist = Mapping.utils.EndPoint(pos, lidar_param, sdata)
    print(sdata)

    # Draw Map
    gmap = GridMap([0.7, -0.9, 5.0, -5.0], gsize=3)
    gmap.update_map(pos, lidar_param, sdata)
    mimg = gmap.adaptive_get_map_prob()
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    mimg_ = cv2.flip(mimg,0)
    cv2.imshow("map", mimg_)

    # Draw Env
    img_ = img.copy()
    for pts in plist:
        cv2.line(
            img_, 
            (int(1*pos[0]), int(1*pos[1])), 
            (int(1*pts[0]), int(1*pts[1])),
            (0.0,1.0,0.0), 1)
    cv2.circle(img_,(pos[0],pos[1]),5,(0.5,0.5,0.5),3)
    img_ = cv2.flip(img_,0)
    cv2.imshow("Mapping",img_)
    k = cv2.waitKey(0)