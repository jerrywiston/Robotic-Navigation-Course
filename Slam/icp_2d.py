import sys
import numpy as np
import cv2
sys.path.append("..")
import Slam.utils as utils

def Align(Xc, Pc):
    # Xc = R * Pc + T
    Pave = np.mean(Pc,0)
    Xave = np.mean(Xc,0)
    Pc = Pc - Pave
    Xc = Xc - Xave

    W = np.matmul(np.transpose(Xc), Pc)
    u, s, vh = np.linalg.svd(W, full_matrices=True)
    R = np.matmul(u,vh)
    T = Xave - np.transpose(np.matmul(R, np.transpose(Pave)))
    return R, T

def Rejection(Xc, Pc):
    error = Xc - Pc
    error = np.sum((error * error),1)
    id_sort = np.argsort(error)
    size = Xc.shape[0]
    min_id = int(size*0.1)
    max_id = int(size*0.9)
    Xc = Xc[id_sort[min_id:max_id]]
    Pc = Pc[id_sort[min_id:max_id]]

    return Xc, Pc

def IcpSolve_scipy(iter, X, P, Rtot=np.eye(2), Ttot=np.zeros((2))):
    from sklearn.neighbors import KDTree
    # X = R * P + T
    if X.shape[0] < 2 or P.shape[0] < 2:
        return np.eye(2), np.zeros((2), dtype=float)

    pc_match = P.copy()
    tree = KDTree(X, leaf_size=2)
    for i in range(iter):
        Pc = utils.Transform(pc_match, Rtot, Ttot)
        Xc = X[tree.query(Pc, k=1)[1]].reshape(Pc.shape)
        Xc, Pc = Rejection(Xc, Pc)
        R, T = Align(Xc, Pc)

        Rtot = np.matmul(R,Rtot)
        Ttot = T + np.matmul(R,Ttot)

    return Rtot, Ttot

def IcpSolve(iter, X, P, Rtot=np.eye(2), Ttot=np.zeros((2))):
    # X = R * P + T
    if X.shape[0] < 2 or P.shape[0] < 2:
        return np.eye(2), np.zeros((2), dtype=float)

    X = X.astype(np.float32)
    pc_match = P.copy().astype(np.float32)
    knn = cv2.ml.KNearest_create()
    response = np.array(range(X.shape[0])).astype(np.float32).reshape(-1,1)
    knn.train(X, cv2.ml.ROW_SAMPLE, response)

    for i in range(iter):
        Pc = utils.Transform(pc_match.astype(np.float32), Rtot, Ttot).astype(np.float32)
        ret, results, neighbours, dist = knn.findNearest(Pc, 1)
        Xc = X[results[:,0].astype(np.int)].squeeze()
        Xc, Pc = Rejection(Xc, Pc)
        R, T = Align(Xc, Pc)

        Rtot = np.matmul(R,Rtot)
        Ttot = T + np.matmul(R,Ttot)

    return Rtot, Ttot

class Icp2dTracking:
    def __init__(self, iteration=30):
        self.iteration = iteration

    def init_tracking(self, observation, start_pose=(0,0,0)):
        self.step_count = 0
        self.timestamp_history = [0.0]
        self.odometry_history = [start_pose]
        self.observation_history = [observation]
        self.rotation = np.eye(2)
        self.translation = np.zeros(2)
    
    def add_observation(self, observation, timestamp=None):
        self.observation_history.append(observation)
        R_pc, T_pc = IcpSolve(self.iteration, observation, self.observation_history[self.step_count])
        R, T = np.transpose(R_pc), -T_pc
        self.rotation, self.translation = utils.TransformRT(R, T, self.rotation, self.translation)
        pose_est = (
            self.odometry_history[0][0] + self.translation[0],
            self.odometry_history[0][1] + self.translation[1],
            self.odometry_history[0][2] + np.rad2deg(np.arctan2(self.rotation[1,0], self.rotation[0,0]))
        )
        self.odometry_history.append(pose_est)
        if timestamp is None:
            self.timestamp_history.append(self.timestamp_history[self.step_count]+1.0)
        else:
            self.timestamp_history.append(timestamp)
        self.step_count += 1
