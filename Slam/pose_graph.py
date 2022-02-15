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

    def compute_A(self, i, j):
        return np.zeros((3,3), dtype=np.float32)

    def compute_B(self, i, j):
        return np.zeros((3,3), dtype=np.float32)

    def compute_e(self, i, j):
        pass

    def compute_jacobian(self):
        n_nodes = len(self.nodes)
        A_len = 3*n_nodes*n_nodes
        jaco = np.zeros((3,A_len*2), dtype=np.float32)
        count = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                jaco[:, 3*count:3*(count+1)] = self.compute_A(i,j)
                jaco[:, A_len+3*count:A_len+3*(count+1)] = self.compute_B(i,j)
                count += 1
        return jaco

    def compute_hassian(self):
        jaco = self.compute_jacobian()
        hass = np.matmul(jaco, np.transpose(jaco))
        return hass
    
    def solve(self):
        pass