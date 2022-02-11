import numpy as np
import cv2

class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, w=0.0):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0
        self.update(x, y, yaw, v, w)
    
    def update(self, x=None, y=None, yaw=None, v=None, w=None):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if yaw is not None:
            self.yaw = yaw
        if v is not None:
            self.v = v
        if w is not None:
            self.w = w

    def pose(self):
        return np.array([self.x, self.y, self.yaw])
    
    def __str__(self):
        return "[State] x={:.4f}, y={:.4f}, yaw={:.4f}, v={:.4f}, w={:.4f}".format(self.x, self.y, self.yaw, self.v, self.w)

class ControlCommand:
    def __init__(self, control_type, *command): 
        #  Support basic/bicycle/wmr
        self.control_type = control_type
        try:
            if control_type == "basic":
                self.v = command[0]
                self.w = command[1]
            elif control_type == "ddv":
                self.lw = command[0]
                self.rw = command[1]
            elif control_type == "bicycle":
                self.a = command[0]
                self.delta = command[1]
            else:
                raise NameError("Unknown control type!!")
        except NameError:
            raise
    
    def __str__(self):
        if self.control_type == "basic":
            return "[Command] v={}, w={}".format(self.v, self.w)
        elif self.control_type == "ddv": # Differential Drive Vehicle
            return "[Command] lw={}, rw={}".format(self.lw, self.rw)
        elif self.control_type == "bicycle":
            return "[Command] a={}, delta={}".format(self.a, self.delta)

def rot_pos(x,y,phi_):
    phi = np.deg2rad(phi_)
    return np.array((x*np.cos(phi)+y*np.sin(phi), -x*np.sin(phi)+y*np.cos(phi)))

def draw_rectangle(img,x,y,u,v,phi,color=(0,0,0),size=1):
    pts1 = rot_pos(-u/2,-v/2,phi) + np.array((x,y))
    pts2 = rot_pos(u/2,-v/2,phi) + np.array((x,y))
    pts3 = rot_pos(-u/2,v/2,phi) + np.array((x,y))
    pts4 = rot_pos(u/2,v/2,phi) + np.array((x,y))
    cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts2.astype(int).tolist()), color, size)
    cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts3.astype(int).tolist()), color, size)
    cv2.line(img, tuple(pts3.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
    cv2.line(img, tuple(pts2.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
    return img

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

def EndPoint(pos, bot_param, sensor_data):
    pts_list = []
    inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
    for i in range(bot_param[0]):
        theta = pos[2] + bot_param[1] + i*inter
        pts_list.append(
            [ pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta)),
              pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta))] )
    return pts_list

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)