import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, find, triu
from itertools import combinations_with_replacement


def rectangle_triangulation(nx, ny):
    idx = np.reshape(np.arange(nx*ny), (ny, nx))
    idx1 = idx[:-1, :-1].flatten()
    idx2 = idx[:-1, 1:].flatten()
    idx3 = idx[1:, 1:].flatten()
    idx4 = idx[1:, :-1].flatten()
    elem1 = np.repeat(idx1, 2)
    elem2 = np.repeat(idx2, 2)
    elem2[1::2] = idx3
    elem3 = np.repeat(idx3, 2)
    elem3[1::2] = idx4
    elem = np.vstack((elem1, elem2, elem3))
    return elem.T


def to_adjacency_matrix(indices_matrix, shape=None):
    n_row, n_col = indices_matrix.shape
    row = np.arange(n_row).repeat(n_col)
    col = indices_matrix.flatten()
    data = np.ones((n_col*n_row,), dtype=int)
    return coo_matrix((data, (row, col)), shape=shape)


def weights_inside_for_PN(N=5):
    k: int = N-3
    comb = list(combinations_with_replacement([1, 2, 3], k))
    comb = np.array(comb)
    comb_with_1 = np.sum(comb == 1, axis=1).reshape((1, -1))
    comb_with_2 = np.sum(comb == 2, axis=1).reshape((1, -1))
    comb_with_3 = np.sum(comb == 3, axis=1).reshape((1, -1))
    weights = np.concatenate((comb_with_1, comb_with_2, comb_with_3), axis=0)
    return (1+weights)/N


class Mesh2d:
    """Triangular 2d mesh with post-processing tools."""

    def __init__(self, node_X, node_Y, elem):
        self.node_X = node_X
        self.node_Y = node_Y
        self.elem = elem
        self.build_adjacency_matrices()

    def build_adjacency_matrices(self):
        self.n_node = self.node_X.shape[0]
        self.n_elem = self.elem.shape[0]

        # sparse adjacency matrix - each row is one element, each column is one node
        # if a node belongs to an element, the corresponding matrix entry is 1; otherwise, it is 0
        self.elem_node_adj = to_adjacency_matrix(self.elem)

        # for each node, calculate number of adjacent elements
        self.n_elements_per_node = np.sum(self.elem_node_adj, axis=0)

        # for each pair of elements, calculate number of common nodes
        # obviously, 3 on diagonal
        self.elem_elem_adj = self.elem_node_adj @ self.elem_node_adj.T

        # for each pair of nodes, calculate number of elements adjacent to both
        # diagional entries show, how many elements are adjacent to that node (n_elements_per_node)
        # if a non-diagonal entry equals 1, these two nodes form a boundary edge
        # if a non-diagonal entry equals 2, these two nodes form an interior edge
        self.node_node_adj = self.elem_node_adj.T @ self.elem_node_adj

        # similarly to elem, this array contains node indices for each edge
        edges_matrix = triu(self.node_node_adj, 1)
        node0, node1, value = find(edges_matrix)
        self.edge = np.concatenate(([node0], [node1])).T
        self.n_edge = self.edge.shape[0]

        # indicates if the edge is on the boundary
        self.edge_boundary_flag = value == 1

        # coordinates of centers of elements:
        self.elem_center_X = np.mean(self.node_X[self.elem], axis=1)
        self.elem_center_Y = np.mean(self.node_Y[self.elem], axis=1)

        # coordinates of midpoints of edges:
        self.edge_mid_X = np.mean(self.node_X[self.edge], axis=1)
        self.edge_mid_Y = np.mean(self.node_Y[self.edge], axis=1)

        # sparse adjacency matrix - each row is one edge, each column is one node
        # if a node belongs to an edge, the corresponding matrix entry is 1; otherwise, it is 0
        self.edge_node_adj = to_adjacency_matrix(self.edge)

        # sparse adjacency matrix - each row is one element, each column is one edge
        # if an edge belongs to an element, the corresponding matrix entry is 2;
        # if they share only 1 node, the entry is 1; otherwise, it is 0
        self.elem_edge_adj_weighted = self.elem_node_adj @ self.edge_node_adj.T

        # sparse adjacency matrix - each row is one element, each column is one edge
        # if an edge belongs to an element, the corresponding matrix entry is 1; otherwise, it is 0
        self.elem_edge_adj = self.elem_edge_adj_weighted.copy()
        self.elem_edge_adj[self.elem_edge_adj_weighted == 1] = 0
        self.elem_edge_adj[self.elem_edge_adj_weighted == 2] = 1
        self.elem_edge_adj.eliminate_zeros()

        # for each element, edge 0 goes from node 0 to 1, 1 (1->2), 2 (2->0)
        # sparse adjacency matrix - each row is one element, each column is one node
        # if the node belongs to i-th side of the element,
        # the corresponding matrix entry is 1; otherwise it is 0
        self.elem_node_adj_partial = []
        for i in range(3):
            elem_edge_i = np.delete(self.elem, i-1, axis=1)
            adj = to_adjacency_matrix(elem_edge_i, shape=(self.n_elem, self.n_node))
            self.elem_node_adj_partial.append(adj)

    def nodes_on_edges_for_PN(self, N=2):
        x = self.node_X[self.edge]
        y = self.node_Y[self.edge]
        # P2: [1/2, 1/2]
        # P3: [1/3, 2/3], [2/3, 1/3]
        # P4: [1/4, 3/4], [2/4, 2/4], [3/4, 1/4]
        # P5: [1/5, 4/5], [2/5, 3/5], [3/5, 2/5], [4/5, 1/5]
        weights0 = np.arange(1, N).reshape((1, -1))/N
        weights = np.concatenate((1-weights0, weights0))
        new_node_X = x @ weights
        new_node_Y = y @ weights
        return new_node_X, new_node_Y

    def nodes_inside_elements_for_PN(self, N=3):
        if N < 3:
            return np.empty((self.n_elem, 0)),  np.empty((self.n_elem, 0))
        # P3: [1/3, 1/3, 1/3]
        # P4: [1/4, 1/4, 2/4], [1/4, 2/4, 1,4]
        # P5: [1/5, 1/5, 3/5], [1/5, 2/5, 2/5], [1/5, 3/5, 1/5], [2/5, 1/5, 2/5], [2/5, 2/5, 1/5], [3/5, 1/5, 1/5]
        if N == 3:
            weights = np.ones((3, 1))/3
        else:
            weights = weights_inside_for_PN(N)
        x = self.node_X[self.elem]
        y = self.node_Y[self.elem]
        new_node_X = x @ weights
        new_node_Y = y @ weights
        return new_node_X, new_node_Y

    def P1_to_PN(self, N):
        # PN node indices for edges
        new_node_X, new_node_Y = self.nodes_on_edges_for_PN(N)
        n_node_per_edge = new_node_X.shape[1]
        n_new_node = n_node_per_edge * self.n_edge
        self.edge_PN = np.arange(self.n_node, self.n_node+n_new_node).reshape((self.n_edge, n_node_per_edge))
        n_node = self.n_node+n_new_node

        # extend "node" matrix
        self.node_X = np.concatenate((self.node_X, new_node_X.flatten()))
        self.node_Y = np.concatenate((self.node_Y, new_node_Y.flatten()))

        self.elem_PN_edge = np.empty((self.n_elem, 0), dtype=int)
        for i in range(3):
            # sparse adjacency matrix - each row is one element, each column is one edge
            # if i-th element side equals this edge, the corresponding matrix entry is 2
            tmp = self.elem_node_adj_partial[i] @ self.edge_node_adj.T
            tmp[tmp == 1] = 0
            tmp.eliminate_zeros()
            _, edge, _ = find(tmp)
            # starting node of i-th edge for each element according to "self.edge" matrix:
            edge_start = self.edge[edge, 0]
            # starting node of i-th edge for each element according to "self.elem" matrix:
            edge_start_correct = self.elem[:, i]
            correct_ordering = edge_start == edge_start_correct

            edge_i = np.zeros((self.n_elem, n_node_per_edge))
            edge_i[correct_ordering, :] = self.edge_PN[edge[correct_ordering]]
            edge_i[~correct_ordering, :] = np.fliplr(self.edge_PN[edge[~correct_ordering]])

            self.elem_PN_edge = np.concatenate((self.elem_PN_edge, edge_i), axis=1)

        # PN node indices for elements - nodes inside
        new_node_X, new_node_Y = self.nodes_inside_elements_for_PN(N)
        n_node_per_elem = new_node_X.shape[1]
        n_new_node = n_node_per_elem * self.n_elem
        self.elem_PN_inside = np.arange(n_node, n_node+n_new_node).reshape((self.n_elem, n_node_per_elem))

        # extend "node" matrix, create "elem_PN" matrix:
        self.node_X = np.concatenate((self.node_X, new_node_X.flatten()))
        self.node_Y = np.concatenate((self.node_Y, new_node_Y.flatten()))
        self.elem_PN = np.concatenate((self.elem, self.elem_PN_edge, self.elem_PN_inside), axis=1)

    def plot_mesh(self):
        plt.triplot(self.node_X, self.node_Y, self.elem)  # plot triangle edges
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Triangular mesh")
        plt.axis('equal')  # ensure equal aspect ratio

    def plot_nodes(self, show_numbers=False, markersize=5):
        plt.plot(self.node_X, self.node_Y, 'o', color="cyan", markersize=markersize)  # plot nodes
        if show_numbers:
            i = 0
            for x, y in zip(self.node_X, self.node_Y):
                plt.text(x, y, str(i))  # , va="center", ha="center")
                i += 1

    def plot_elem_centers(self, show_numbers=False, markersize=3):
        plt.plot(self.elem_center_X, self.elem_center_Y, 'o', color='orange', markersize=markersize)
        if show_numbers:
            i = 0
            for x, y in zip(self.elem_center_X, self.elem_center_Y):
                plt.text(x, y, str(i), va="center", ha="center")
                i += 1

    def plot_edge_midpoints(self, show_numbers=False, markersize=3):
        plt.plot(self.edge_mid_X, self.edge_mid_Y, 'o',  color="green", markersize=markersize)
        if show_numbers:
            i = 0
            for x, y in zip(self.edge_mid_X, self.edge_mid_Y):
                plt.text(x, y, str(i))
                i += 1


class Mesh3d:
    """Triangular 3d mesh post-processing tools."""

    def __init__(self, node_X, node_Y, node_Z, elem, face):
        self.node_X = node_X
        self.node_Y = node_Y
        self.node_Z = node_Z
        self.elem = elem
        self.face = face
        self.face_PN = np.empty((0, 0))
        self.build_adjacency_matrices()

    def build_adjacency_matrices(self):
        self.n_node = self.node_X.shape[0]
        self.n_elem = self.elem.shape[0]
        self.n_face = self.face.shape[0]

        # sparse adjacency matrix - each row is one element, each column is one node
        # if a node belongs to an element, the corresponding matrix entry is 1; otherwise, it is 0
        self.elem_node_adj = to_adjacency_matrix(self.elem)

        # for each node, calculate number of adjacent elements
        self.n_elements_per_node = np.sum(self.elem_node_adj, axis=0)

        # for each pair of elements, calculate number of common nodes
        # obviously, 4 on diagonal
        self.elem_elem_adj = self.elem_node_adj @ self.elem_node_adj.T

        # for each pair of nodes, calculate number of elements adjacent to both
        # diagional entries show, how many elements are adjacent to that node (n_elements_per_node)
        self.node_node_adj = self.elem_node_adj.T @ self.elem_node_adj

        # similarly to elem, this array contains node indices for each edge
        edges_matrix = triu(self.node_node_adj, 1)
        node0, node1, value = find(edges_matrix)
        self.edge = np.concatenate(([node0], [node1])).T
        self.n_edge = self.edge.shape[0]

        # indicates if the edge is on the boundary
        self.edge_boundary_flag = value == 1

        # sparse adjacency matrix - each row is one face, each column is one node
        # if a node belongs to a face, the corresponding matrix entry is 1; otherwise, it is 0
        self.face_node_adj = to_adjacency_matrix(self.face)
        # TODO: If needed, identify faces on boundary. Probably can be exported from gmsh.

        # coordinates of centers of elements:
        self.elem_center_X = np.mean(self.node_X[self.elem], axis=1)
        self.elem_center_Y = np.mean(self.node_Y[self.elem], axis=1)
        self.elem_center_Z = np.mean(self.node_Z[self.elem], axis=1)

        # coordinates of midpoints of edges:
        self.edge_mid_X = np.mean(self.node_X[self.edge], axis=1)
        self.edge_mid_Y = np.mean(self.node_Y[self.edge], axis=1)
        self.edge_mid_Z = np.mean(self.node_Z[self.edge], axis=1)

        # sparse adjacency matrix - each row is one edge, each column is one node
        # if a node belongs to an edge, the corresponding matrix entry is 1; otherwise, it is 0
        self.edge_node_adj = to_adjacency_matrix(self.edge)

        # sparse adjacency matrix - each row is one element, each column is one edge
        # if an edge belongs to an element, the corresponding matrix entry is 2;
        # if they share only 1 node, the entry is 1; otherwise, it is 0
        self.elem_edge_adj_weighted = self.elem_node_adj @ self.edge_node_adj.T

        # sparse adjacency matrix - each row is one element, each column is one edge
        # if an edge belongs to an element, the corresponding matrix entry is 1; otherwise, it is 0
        self.elem_edge_adj = self.elem_edge_adj_weighted.copy()
        self.elem_edge_adj[self.elem_edge_adj_weighted == 1] = 0
        self.elem_edge_adj[self.elem_edge_adj_weighted == 2] = 1
        self.elem_edge_adj.eliminate_zeros()

        # for each element, edge 0 goes from node 0 to 1, 1 (1->2), 2 (2->0)
        # sparse adjacency matrix - each row is one element, each column is one node
        # if the node belongs to i-th side of the element,
        # the corresponding matrix entry is 1; otherwise it is 0
        self.elem_node_adj_partial = []
        for i in range(3):
            elem_edge_i = np.delete(self.elem[:, :3], i-1, axis=1)
            adj = to_adjacency_matrix(elem_edge_i, shape=(self.n_elem, self.n_node))
            self.elem_node_adj_partial.append(adj)
        # for each element, edge 3 goes from node 0 to 3, 4 (1->3), 5 (2->3)
        for i in range(3):
            elem_edge_i = np.concatenate((self.elem[:, i].reshape((-1, 1)), self.elem[:, 3].reshape(-1, 1)), axis=1)
            adj = to_adjacency_matrix(elem_edge_i, shape=(self.n_elem, self.n_node))
            self.elem_node_adj_partial.append(adj)

        # TODO: add comment - FACES ON BOUNDARY
        self.face_node_adj_partial = []
        for i in range(3):
            face_edge_i = np.delete(self.face[:, :3], i-1, axis=1)
            adj = to_adjacency_matrix(face_edge_i, shape=(self.n_face, self.n_node))
            self.face_node_adj_partial.append(adj)

    def nodes_on_edges_for_PN(self, N=2):
        x = self.node_X[self.edge]
        y = self.node_Y[self.edge]
        z = self.node_Z[self.edge]
        # P2: [1/2, 1/2]
        # P3: [1/3, 2/3], [2/3, 1/3]
        # P4: [1/4, 3/4], [2/4, 2/4], [3/4, 1/4]
        # P5: [1/5, 4/5], [2/5, 3/5], [3/5, 2/5], [4/5, 1/5]
        weights0 = np.arange(1, N).reshape((1, -1))/N
        weights = np.concatenate((1-weights0, weights0))
        new_node_X = x @ weights
        new_node_Y = y @ weights
        new_node_Z = z @ weights
        return new_node_X, new_node_Y, new_node_Z

    def nodes_inside_elements_for_PN(self, N=3):
        # TODO: for N>3, nodes inside faces; for N>4 also nodes inside elements
        if N < 3:
            return np.empty((self.n_elem, 0)),  np.empty((self.n_elem, 0))
        # P3: [1/3, 1/3, 1/3]
        # P4: [1/4, 1/4, 2/4], [1/4, 2/4, 1,4]
        # P5: [1/5, 1/5, 3/5], [1/5, 2/5, 2/5], [1/5, 3/5, 1/5], [2/5, 1/5, 2/5], [2/5, 2/5, 1/5], [3/5, 1/5, 1/5]
        if N == 3:
            weights = np.ones((3, 1))/3
        else:
            weights = weights_inside_for_PN(N)
        x = self.node_X[self.elem]
        y = self.node_Y[self.elem]
        new_node_X = x @ weights
        new_node_Y = y @ weights
        return new_node_X, new_node_Y

    def P1_to_PN(self, N):
        if N > 2:
            print("P1_to_PN function not implemented for N>2")
            return None
        # PN node indices for edges
        new_node_X, new_node_Y, new_node_Z = self.nodes_on_edges_for_PN(N)
        n_node_per_edge = new_node_X.shape[1]
        n_new_node = n_node_per_edge * self.n_edge
        self.edge_PN = np.arange(self.n_node, self.n_node+n_new_node).reshape((self.n_edge, n_node_per_edge))
        n_node = self.n_node+n_new_node

        # extend "node" matrix
        self.node_X = np.concatenate((self.node_X, new_node_X.flatten()))
        self.node_Y = np.concatenate((self.node_Y, new_node_Y.flatten()))
        self.node_Z = np.concatenate((self.node_Z, new_node_Z.flatten()))

        self.elem_PN_edge = np.empty((self.n_elem, 0), dtype=int)
        for i in range(6):
            # sparse adjacency matrix - each row is one element, each column is one edge
            # if i-th element side equals this edge, the corresponding matrix entry is 2
            tmp = self.elem_node_adj_partial[i] @ self.edge_node_adj.T
            tmp[tmp == 1] = 0
            tmp.eliminate_zeros()
            _, edge, _ = find(tmp)
            # starting node of i-th edge for each element according to "self.edge" matrix:
            edge_start = self.edge[edge, 0]
            # starting node of i-th edge for each element according to "self.elem" matrix:
            edge_start_correct = self.elem[:, min(i, 3)]
            correct_ordering = edge_start == edge_start_correct

            edge_i = np.zeros((self.n_elem, n_node_per_edge))
            edge_i[correct_ordering, :] = self.edge_PN[edge[correct_ordering]]
            edge_i[~correct_ordering, :] = np.fliplr(self.edge_PN[edge[~correct_ordering]])

            self.elem_PN_edge = np.concatenate((self.elem_PN_edge, edge_i), axis=1)

        # FACES ON THE BOUNDARY:
        self.face_PN_edge = np.empty((self.n_face, 0), dtype=int)
        for i in range(3):
            # sparse adjacency matrix - each row is one face, each column is one edge
            # if i-th face side equals this face, the corresponding matrix entry is 2
            tmp = self.face_node_adj_partial[i] @ self.edge_node_adj.T
            tmp[tmp == 1] = 0
            tmp.eliminate_zeros()
            _, edge, _ = find(tmp)
            edge_start = self.edge[edge, 0]
            # starting node of i-th edge for each element according to "self.elem" matrix:
            edge_start_correct = self.face[:, min(i, 3)]
            correct_ordering = edge_start == edge_start_correct

            edge_i = np.zeros((self.n_face, n_node_per_edge))
            edge_i[correct_ordering, :] = self.edge_PN[edge[correct_ordering]]
            edge_i[~correct_ordering, :] = np.fliplr(self.edge_PN[edge[~correct_ordering]])

            self.face_PN_edge = np.concatenate((self.face_PN_edge, edge_i), axis=1)

        # PN node indices for elements - nodes inside
        # TODO: nodes_inside_elements_for_PN
        """
        new_node_X, new_node_Y = self.nodes_inside_elements_for_PN(N)
        n_node_per_elem = new_node_X.shape[1]
        n_new_node = n_node_per_elem * self.n_elem
        self.elem_PN_inside = np.arange(n_node, n_node+n_new_node).reshape((self.n_elem, n_node_per_elem))

        # extend "node" matrix, create "elem_PN" matrix:
        self.node_X = np.concatenate((self.node_X, new_node_X.flatten()))
        self.node_Y = np.concatenate((self.node_Y, new_node_Y.flatten()))
        """

        self.elem_PN = np.concatenate((self.elem, self.elem_PN_edge), axis=1)
        self.face_PN = np.concatenate((self.face, self.face_PN_edge), axis=1)
        # TODO: self.elem_PN = np.concatenate((self.elem, self.elem_PN_edge, self.elem_PN_inside), axis=1)

    def plot_mesh(self):
        ax = plt.axes(projection='3d')
        for i in range(4):
            triangles = np.delete(self.elem, i, axis=1)
            ax.plot_trisurf(self.node_X, self.node_Y, self.node_Z, triangles=triangles, linewidths=0.2, edgecolor='k', color=[0, 1.0, 0.5, 0.05])
        ax.plot_trisurf(self.node_X, self.node_Y, self.node_Z, triangles=self.face, linewidths=0.2, edgecolor='k', color=[0, 1.0, 0.5, 0.05])
        ax.set_aspect('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Triangular mesh")
        plt.axis('equal')  # ensure equal aspect ratio
        return ax

    def plot_nodes(self, ax, show_numbers=False, markersize=5):
        ax.plot(self.node_X, self.node_Y, self.node_Z, 'o', color="cyan", markersize=markersize)  # plot nodes
        if show_numbers:
            i = 0
            for x, y, z in zip(self.node_X, self.node_Y, self.node_Z):
                ax.text(x, y, z, str(i))  # , va="center", ha="center")
                i += 1

    def plot_elem_centers(self, ax, show_numbers=False, markersize=3):
        ax.plot(self.elem_center_X, self.elem_center_Y, self.elem_center_Z, 'o', color='orange', markersize=markersize)
        if show_numbers:
            i = 0
            for x, y, z in zip(self.elem_center_X, self.elem_center_Y, self.elem_center_Z):
                ax.text(x, y, z, str(i), va="center", ha="center")
                i += 1

    def plot_edge_midpoints(self, ax, show_numbers=False, markersize=3):
        ax.plot(self.edge_mid_X, self.edge_mid_Y, self.edge_mid_Z, 'o',  color="green", markersize=markersize)
        if show_numbers:
            i = 0
            for x, y, z in zip(self.edge_mid_X, self.edge_mid_Y, self.edge_mid_Z):
                ax.text(x, y, z, str(i))
                i += 1
