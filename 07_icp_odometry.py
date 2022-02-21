import argparse
import numpy as np
import cv2
from Simulation.utils import ControlState
from Simulation.simulator_map import SimulatorMapLidar
import Slam.utils as utils
from Slam.icp_2d import Icp2dTracking 

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
    pts = np.array(utils.EndPoint((0,0,0), lidar_params, info["lidar"], True))
    # Init Icp
    icp_tracking = Icp2dTracking()
    icp_tracking.init_tracking(pts, start_pose)

    # Init Ground Truth Rot, Trans
    R_acc_gt = np.eye(2)
    T_acc_gt = np.zeros(2)
    odo_gt_hist = [simulator.state.pose()]
    
    count = 0
    pose_hist = [simulator.state.pose()]
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
            pose = simulator.state.pose()
            pts = np.array(utils.EndPoint((0,0,0), lidar_params, info["lidar"], True))
            
            # GT Test
            yaw_diff = np.deg2rad(pose[2] - pose_hist[-1][2])
            R_gt = np.array([[np.cos(yaw_diff), -np.sin(yaw_diff)],[np.sin(yaw_diff), np.cos(yaw_diff)]])
            T_gt = np.array((pose[0] - pose_hist[-1][0], pose[1] - pose_hist[-1][1]))
            T_gt = np.transpose(np.matmul(np.transpose(np.matmul(R_gt,R_acc_gt)), np.transpose(T_gt)))
            R_acc_gt, T_acc_gt = utils.TransformRT(R_gt, T_gt, R_acc_gt, T_acc_gt)
            odo_gt_hist.append((start_pose[0]+T_acc_gt[0], start_pose[1]+T_acc_gt[1]))
            pose_hist.append(pose)

            # ICP
            error = icp_tracking.add_observation(pts, count*simulator.dt)

        # Draw Estimate Path
        odo_hist = icp_tracking.odometry_history
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
    img = cv2.flip(cv2.imread("Maps/map1.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.

    run_odometry(m)
