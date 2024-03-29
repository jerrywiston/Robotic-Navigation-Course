import numpy as np
import cv2
from Simulation.sensor_lidar import LidarModel
import Slam.utils
from Slam.grid_map import GridMap 

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
    pose = (100,200,0)
    sdata = lidar.measure(img, pose)
    plist = Slam.utils.EndPoint(pose, lidar_param, sdata)
    print(sdata)

    # Draw Map
    gmap = GridMap([0.7, -0.9, 5.0, -5.0], gsize=3)
    gmap.update_map(pose, lidar_param, sdata)
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
            (int(1*pose[0]), int(1*pose[1])), 
            (int(1*pts[0]), int(1*pts[1])),
            (0.0,1.0,0.0), 1)
    cv2.circle(img_,(pose[0],pose[1]),5,(0.5,0.5,0.5),3)
    img_ = cv2.flip(img_,0)
    cv2.imshow("Mapping",img_)
    k = cv2.waitKey(0)