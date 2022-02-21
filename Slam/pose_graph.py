import numpy as np

class Node:
    def __init__(self, pose):
        self.data = pose
        self.in_edges = []
        self.out_edges = []

    def transform(self):
        R = np.eye(2, dtype=np.float32)
        R[0,0] = np.cos(np.deg2rad(self.pose[2]))
        R[0,1] = np.sin(np.deg2rad(self.pose[2]))
        R[1,0] = -np.sin(np.deg2rad(self.pose[2]))
        R[1,1] = np.cos(np.deg2rad(self.pose[2]))
        T = np.zeros((2,1), dtype=np.float32)
        T[0,0] = self.pose[0]
        T[1,0] = self.pose[1]
        return R, T

class Edge:
    def __init__(self, id1, id2, pose):
        self.id1 = id1
        self.id2 = id2
        self.data = pose
    
    def transform(self):
        R = np.eye(2, dtype=np.float32)
        R[0,0] = np.cos(np.deg2rad(self.pose[2]))
        R[0,1] = np.sin(np.deg2rad(self.pose[2]))
        R[1,0] = -np.sin(np.deg2rad(self.pose[2]))
        R[1,1] = np.cos(np.deg2rad(self.pose[2]))
        T = np.zeros((2,1), dtype=np.float32)
        T[0,0] = self.pose[0]
        T[1,0] = self.pose[1]
        return R, T

    def get_A(self):
        A = np.zeros((3,3), dtype=np.float32)
        Ri, Ti = self.nodes[self.id1].transform()
        A[:2,:2] = -np.transpose(Ri)
        Ridtheta = np.array([[Ri[1,0],-Ri[0,0]],\
                            [Ri[0,0],Ri[1,0]]])
        diff_temp = np.array([[self.nodes[self.id2][0]-self.nodes[self.id1][0]],\
                            [self.nodes[self.id2][1]-self.nodes[self.id1][1]]])
        A[:2,2] = np.matmul(Ridtheta, diff_temp)
        A[2,2] = -1 
        return A

    def get_B(self):
        B = np.zeros((3,3), dtype=np.float32)
        Ri, Ti = self.nodes[self.id1].transform()
        B[:2,:2] = np.transpose(Ri)
        B[2,2] = -1
        return B

    def get_e(self):
        e = np.zeros((3,1), dtype=np.float32)
        Ri, Ti = self.nodes[self.id1].transform()
        diff_temp = np.array([[self.nodes[self.id2][0]-self.nodes[self.id1][0]],\
                            [self.nodes[self.id2][1]-self.nodes[self.id1][1]]])
        e[:2,0] = np.matmul(np.transpose(Ri), diff_temp)
        e[2,0] = np.deg2rad((self.nodes[self.id2].pose[2] - self.nodes[self.id1].pose[2] - self.pose[2]) % 360)
        return e

class PoseGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.error = None

    def add_node(self, pose):
        node = Node(pose)
        self.nodes.append(node)
        node_id = len(self.nodes)-1
        return node_id
    
    def add_edge(self, id1, id2, pose):
        edge = Edge(id1, id2, pose)
        self.edges.append(edge)
        edge_id = len(self.edges)-1
        self.nodes[id1].out_edges.append(edge_id)
        self.nodes[id2].in_edges.append(edge_id)
        return edge_id

    def compute_jacobian(self):
        n_nodes = len(self.nodes)
        A_len = 3*n_nodes*n_nodes
        jaco = np.zeros((3,A_len*2), dtype=np.float32)
        for edge in self.edges:
            i, j = edge.id1, edge.id2
            jaco[:,3*(i*n_nodes+j):3*(i*n_nodes+j+1)] = edge.get_A()
            jaco[:, A_len+3*(i*n_nodes+j):A_len+3*(i*n_nodes+j+1)] = edge.get_B()
        return jaco

    def compute_err(self):
        n_nodes = len(self.nodes)
        err = np.zeros((3,3*n_nodes*n_nodes), dtype=np.float32)
        for edge in self.edges:
            i, j = edge.id1, edge.id2
            err[:,3*(i*n_nodes+j):3*(i*n_nodes+j+1)] = edge.get_e()
        return err

    def solve(self):
        jaco = self.compute_jaco()
        H = np.matmul(np.transpose(jaco), jaco)
        H[:3,:3] = np.eye(3,dtype=np.float32)
        err = self.compute_err()
        b = np.matmul(np.transpose(jaco), err)
        # H delta = -b
        # psudo inverse: Ax=b -> x = (A^T A)^(-1) A^T b
        temp = np.linalg.inv(np.matmul(np.transpose(H), H))
        delta = np.matmul(np.matmul(temp, np.transpose(H)), -b)
        return delta