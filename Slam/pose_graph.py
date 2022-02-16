import numpy as np

class Node:
    def __init__(self, pose):
        self.data = pose
        self.in_edges = []
        self.out_edges = []

class Edge:
    def __init__(self, id1, id2, pose):
        self.id1 = id1
        self.id2 = id2
        self.data = pose

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

    def get_A(self, i, j):
        return np.zeros((3,3), dtype=np.float32)

    def get_B(self, i, j):
        return np.zeros((3,3), dtype=np.float32)

    def get_e(self, i, j):
        pass

    def compute_jacobian(self):
        n_nodes = len(self.nodes)
        A_len = 3*n_nodes*n_nodes
        jaco = np.zeros((3,A_len*2), dtype=np.float32)
        count = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                jaco[:, 3*count:3*(count+1)] = self.get_A(i,j)
                jaco[:, A_len+3*count:A_len+3*(count+1)] = self.get_B(i,j)
                count += 1
        return jaco

    def compute_err(self):
        n_nodes = len(self.nodes)
        err = np.zeros((3,3*n_nodes*n_nodes), dtype=np.float32)
        count = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                err[:, 3*count:3*(count+1)] = self.get_e(i,j)
                count += 1

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