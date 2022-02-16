import numpy as np
import cv2

# https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#Python
def Bresenham(x0, x1, y0, y1):
    rec = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            rec.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            rec.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return rec

def EndPoint(pose, lidar_params, sensor_data, skip_max=False):
    pts_list = []
    inter = (lidar_params[2] - lidar_params[1]) / (lidar_params[0]-1)
    for i in range(int(lidar_params[0])):
        if skip_max and sensor_data[i] == lidar_params[3]:
            continue
        theta = pose[2] + lidar_params[1] + i*inter
        pts_list.append(
            [ pose[0]+sensor_data[i]*np.cos(np.deg2rad(theta)),
              pose[1]+sensor_data[i]*np.sin(np.deg2rad(theta))] )
    return pts_list

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def Transform(X, R, T):
    Xt = np.transpose(np.matmul(R, np.transpose(X)))
    for i in range(Xt.shape[0]):
        Xt[i] += T
    return Xt

def TransformRT(R, T, R_acc, T_acc):
    R_new = np.matmul(R, R_acc)
    T_new = np.transpose(np.matmul(R_new, T)) + T_acc
    return R_new, T_new