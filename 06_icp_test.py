import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

from Simulation.utils import State, ControlState
from Simulation.simulator_basic import SimulatorBasic
from Simulation.simulator_map import SimulatorMapLidar
import Slam.utils as utils
import Slam.icp_2d as icp

if __name__ == "__main__":
    # Read Map
    img = cv2.flip(cv2.imread("Maps/map1.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    
    # Initialize Simulator
    start_pose = (100,200,0)
    lidar_params = [241,-120.0,120.0,400.0]
    simulator = SimulatorMapLidar(SimulatorBasic, m, lidar_params, v_range=50)
    simulator.init_pose(start_pose)
    
    # Step 1
    state_next, info = simulator.step(None)
    print(simulator.state)
    sdata1 = info["lidar"].copy()
    img1 = cv2.flip(simulator.render(), 0)
    cv2.imshow("render1", img1)
    pts_list1 = utils.EndPoint((0,0,0), lidar_params, sdata1, True)
    pts_list1 = np.array(pts_list1)
    plt.figure()
    plt.axis("equal")
    plt.plot(pts_list1[:,0], pts_list1[:,1], "bo")
    
    # Step 2
    for i in range(10):
        state_next, info = simulator.step(ControlState("basic", 20, 20))
        print(simulator.state)
    sdata2 = info["lidar"].copy()
    img2 = cv2.flip(simulator.render(), 0)
    cv2.imshow("render2", img2)

    pts_list2 = utils.EndPoint((0,0,0), lidar_params, sdata2, True)
    pts_list2 = np.array(pts_list2)
    plt.axis("equal")
    plt.plot(pts_list2[:,0], pts_list2[:,1], "go")

    # Iterative Closest Points
    pts_origin = utils.EndPoint((0,0,0), lidar_params, sdata1, True)
    pts_origin = np.array(pts_origin)
    pts_align = utils.EndPoint((0,0,0), lidar_params, sdata2, True)
    pts_align = np.array(pts_align)
    start = time.time()
    R, T = icp.IcpSolve(30, pts_origin, pts_align)
    end = time.time()
    print(end-start)
    print(R,T)
    print(simulator.state)
    
    pts_align_trans = utils.Transform(pts_align, R, T)
    plt.figure()
    plt.axis("equal")
    plt.plot(pts_origin[:,0], pts_origin[:,1], "bo")
    plt.plot(pts_align_trans[:,0], pts_align_trans[:,1], "go")
    
    cv2.waitKey(0)
    plt.show()
    