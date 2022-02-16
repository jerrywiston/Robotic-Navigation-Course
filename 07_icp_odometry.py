import argparse
import numpy as np
import cv2
from Simulation.utils import ControlState
from Simulation.simulator_map import SimulatorMapLidar
from Slam.utils import *
from Slam.icp_2d import *

# Basic Kinematic Model
def run_odometry(m):
    simulator_name = "basic"
    from Simulation.simulator_basic import SimulatorBasic
    print("Control Hint:")
    print("[W] Increase velocity.")
    print("[S] Decrease velocity.")
    print("[A] Increase angular velocity. (Anti-Clockwise)")
    print("[D] Decrease angular velocity. (Clockwise)")
    print("====================")
    lidar_params = [241,-120.0,120.0,400.0]
    simulator = SimulatorMapLidar(SimulatorBasic, m, lidar_params)
    start_pose = (100.0,200.0,0.0)
    _, info = simulator.init_pose(start_pose)
    
    # Init Rot, Trans
    R_acc = np.eye(2)
    T_acc = np.zeros(2)
    odo_hist = [simulator.state.pose()]
    R_acc_gt = np.eye(2)
    T_acc_gt = np.zeros(2)
    odo_gt_hist = [simulator.state.pose()]
    
    count = 0
    pose_hist = [simulator.state.pose()]
    lidar_hist = [np.array(EndPoint((0,0,0), lidar_params, info["lidar"], True))]

    while(True):
        count += 1
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlState(simulator_name, None, simulator.cstate.w+5)
        elif k == ord("d"):
            command = ControlState(simulator_name, None, simulator.cstate.w-5)
        elif k == ord("w"):
            command = ControlState(simulator_name, simulator.cstate.v+4, None)
        elif k == ord("s"):
            command = ControlState(simulator_name, simulator.cstate.v-4, None)
        elif k == 27:
            print()
            break
        else:
            command = ControlState(simulator_name, None, None)
        _, info = simulator.step(command)
        print("\r", simulator, end="\t")
        img = simulator.render()
        
        if count % 10 == 0:
            count = 0
            pose = simulator.state.pose()
            pts = np.array(EndPoint((0,0,0), lidar_params, info["lidar"], True))
            
            # GT Test
            yaw_diff = np.deg2rad(pose[2] - pose_hist[-1][2])
            R_gt = np.array([[np.cos(yaw_diff), -np.sin(yaw_diff)],[np.sin(yaw_diff), np.cos(yaw_diff)]])
            T_gt = np.array((pose[0] - pose_hist[-1][0], pose[1] - pose_hist[-1][1]))
            T_gt = np.transpose(np.matmul(np.transpose(np.matmul(R_gt,R_acc_gt)), np.transpose(T_gt)))
            R_acc_gt, T_acc_gt = TransformRT(R_gt, T_gt, R_acc_gt, T_acc_gt)
            odo_gt_hist.append((start_pose[0]+T_acc_gt[0], start_pose[1]+T_acc_gt[1]))
            pose_hist.append(pose)

            # ICP
            R_inv,T_inv = Icp(30, pts, lidar_hist[-1])
            R, T = np.transpose(R_inv), -T_inv
            #
            R_acc, T_acc = TransformRT(R, T, R_acc, T_acc)
            lidar_hist.append(pts)
            odo_hist.append((start_pose[0]+T_acc[0], start_pose[1]+T_acc[1]))

        # Draw Estimate Path
        for i, p in enumerate(odo_hist):
            if i != 0:
                cv2.line(img, (int(p[0]), int(p[1])), (int(odo_hist[i-1][0]), int(odo_hist[i-1][1])), (1,0,0), 1)
            cv2.circle(img, (int(p[0]), int(p[1])), 2, (1,0,0))
        for i, p in enumerate(odo_gt_hist):
            cv2.circle(img, (int(p[0]), int(p[1])), 2, (0,0,1))
        img = cv2.flip(img, 0)
        cv2.imshow("ICP Odometry", img)
        
if __name__ == "__main__":
    # Read Map
    img = cv2.flip(cv2.imread("Maps/map3.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.

    run_odometry(m)
