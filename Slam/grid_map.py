import numpy as np
import Slam.utils as utils

class GridMap:
    def __init__(self, map_param, gsize=3.0):
        self.map_param = map_param
        self.map_size = (1000,1000)
        self.gmap = np.zeros(self.map_size,dtype=np.float)
        self.gsize = gsize
        self.boundary = [9999,-9999,9999,-9999]

    def get_grid_prob(self, pos, scale=False):
        if scale:
            pos_grid = (int(pos[0]/self.gsize), int(pos[1]/self.gsize))
        else:
            pos_grid = (int(pos[0]), int(pos[1]))
        pos_grid = (int(pos[0]), int(pos[1]))

        if pos_grid[0] >= self.map_size[1] or pos_grid[0] < 0:
            return 0.5
        if pos_grid[1] >= self.map_size[0] or pos_grid[1] < 0:
            return 0.5
        return self.gmap[pos_grid[1],pos_grid[0]]

    def get_map_prob(self, x0, x1, y0, y1):
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        crop_gmap = self.gmap[y0:y1,x0:x1]
        return np.exp(crop_gmap) / (1.0 + np.exp(crop_gmap))

    def adaptive_get_map_prob(self):
        mimg = self.get_map_prob(
            self.boundary[0], self.boundary[1], 
            self.boundary[2], self.boundary[3] )
        return mimg

    def map_line(self, x0, x1, y0, y1, hit):
        # Scale the position
        x0, x1 = int(x0/self.gsize), int(x1/self.gsize)
        y0, y1 = int(y0/self.gsize), int(y1/self.gsize)

        rec = utils.Bresenham(x0, x1, y0, y1)
        change_list1 = []
        change_list2 = []
        for i in range(len(rec)):
            if i < len(rec)-3 or not hit:
                change_list1.append([rec[i][1],rec[i][0]])
            else:
                change_list2.append([rec[i][1],rec[i][0]])
            
            if rec[i][0] < self.boundary[0]:
                self.boundary[0] = rec[i][0]
            elif rec[i][0] > self.boundary[1]:
                self.boundary[1] = rec[i][0]
            if rec[i][1] < self.boundary[2]:
                self.boundary[2] = rec[i][1]
            elif rec[i][1] > self.boundary[3]:
                self.boundary[3] = rec[i][1]
            
        c1 = np.array(change_list1).tolist()
        c2 = np.array(change_list2).tolist()
        #self.gmap[c1] += self.map_param[0]
        #self.gmap[c2] += self.map_param[1]
        return c1, c2
    
    def update_map(self, bot_pos, bot_param, sensor_data):
        inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
        c1_list, c2_list = [], []
        for i in range(bot_param[0]):
            if sensor_data[i] > bot_param[3] or sensor_data[i] < 1:
                continue
            theta = bot_pos[2] + bot_param[1] + i*inter
            hit = True
            if sensor_data[i] == bot_param[3]:
                hit = False
            c1, c2 = self.map_line(
                int(bot_pos[0]), 
                int(bot_pos[0]+sensor_data[i]*np.cos(np.deg2rad(theta))),
                int(bot_pos[1]),
                int(bot_pos[1]+sensor_data[i]*np.sin(np.deg2rad(theta))),
                hit
            )
            c1_list += c1
            c2_list += c2
        self.gmap[tuple(np.array(c1_list).T)] += self.map_param[0]
        self.gmap[tuple(np.array(c2_list).T)] += self.map_param[1]
