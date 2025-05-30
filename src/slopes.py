import sys
import gmsh
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import h5py
import mesh_tools


def box_from_coordinates(coordinates):
    points = []
    for i in range(8):
        points.append(factory.add_point(*coordinates[i]))
    lines = []
    for i in range(4):
        lines.append(factory.add_line(points[i], points[(i+1) % 4]))
    for i in range(4):
        lines.append(factory.add_line(points[i+4], points[(i+1) % 4+4]))
    for i in range(4):
        lines.append(factory.add_line(points[i], points[i+4]))
    surfaces = []
    loops = [[0,1,2,3],[4,5,6,7],[0,8,4,9],[1,9,5,10],[2,10,6,11],[3,11,7,8]]
    for loop in loops:
        line_tags = [lines[i] for i in loop]
        loop = factory.add_curve_loop(line_tags)
        surfaces.append(factory.add_plane_surface([loop]))
    surface_loop = factory.add_surface_loop(surfaces)
    return factory.add_volume([surface_loop])


gmsh.initialize()
factory = gmsh.model.occ

h = 4.0
x_sum = 50.0
z_sum = 18.3
z_below_layer = 4.5
z_down = 6.1
layer_width = 0.5
x_up = 18.3
x_down = 20.0
y_sum = 2  # thickness of the domain

A = factory.add_point(0, 0, 0)
B = factory.add_point(x_sum, 0, 0)
C = factory.add_point(x_sum, 0, z_down)
D = factory.add_point(x_sum-x_down, 0, z_down)
E = factory.add_point(x_up, 0, z_sum)
F = factory.add_point(0, 0, z_sum)

# Boundary segments:
AB = factory.add_line(A, B)
BC = factory.add_line(B, C)
CD = factory.add_line(C, D)
DE = factory.add_line(D, E)
EF = factory.add_line(E, F)
FA = factory.add_line(F, A)

ABCDEF = factory.add_curve_loop([AB, BC, CD, DE, EF, FA])
whole_slope = factory.add_plane_surface([ABCDEF])

# Thin layer (rectangle):
LA = factory.add_point(0, 0, z_below_layer)
LB = factory.add_point(x_sum, 0, z_below_layer)
LC = factory.add_point(x_sum, 0, z_below_layer+layer_width)
LD = factory.add_point(0, 0, z_below_layer+layer_width)
LAB = factory.add_line(LA, LB)
LBC = factory.add_line(LB, LC)
LCD = factory.add_line(LC, LD)
LDA = factory.add_line(LD, LA)
LABCD = factory.add_curve_loop([LAB, LBC, LCD, LDA])
thin_layer = factory.add_plane_surface([LABCD])

# Three subdomains:
factory.cut([(2, whole_slope)], [(2, thin_layer)], removeTool=False)

# Extrude to 3d:
factory.extrude(factory.get_entities(2), 0, y_sum, 0)

# Synchronize and generate mesh:
factory.remove_all_duplicates()
factory.synchronize()
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
gmsh.model.mesh.generate()
gmsh.model.mesh.remove_duplicate_nodes()
gmsh.model.mesh.remove_duplicate_elements()

# Export the Gmsh mesh to a file
gmsh.write("mesh_output.msh")

# Write mesh data to file:
# gmsh.write("GFG.msh")

# Create graphical user interface
# if 'close' not in sys.argv:
#     gmsh.fltk.run()

# Get node coordinates:
nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()

# Get (triangular) faces:
faceTypes, faceTags, faceNodeTags = gmsh.model.mesh.getElements(dim=2)

# Get (tetrahedral) elements:
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=3)

# Extract X-, Y-, and Z- coordinates of all nodes from GMSH outputs:
node = np.array(nodeCoords)
node_X = node[0::3]
node_Y = node[1::3]
node_Z = node[2::3]

# All (tetrahedral) elements:
elem = elemNodeTags[0].reshape(-1, 4)
elem = elem-1  # indexing from 0

# All (triangular) faces on the boundary (including subdomains):
face = faceNodeTags[0].reshape(-1, 3)
face = face-1  # indexing from 0

# Subdomain tags for each element:
n_elem = elem.shape[0]
min_elemTag = min(elemTags[0])
material = np.zeros((n_elem,), dtype=int)
for i in range(3):
    _, tags, _ = gmsh.model.mesh.getElements(dim=3, tag=i+1)
    material[tags[0]-min_elemTag] = i+1


# Get node coordinates:
nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes()

# Get (triangular) faces:
faceTypes, faceTags, faceNodeTags = gmsh.model.mesh.getElements(dim=2)

# Get (tetrahedral) elements:
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=3)

# Extract X-, Y-, and Z- coordinates of all nodes from GMSH outputs:
node = np.array(nodeCoords)
node_X = node[0::3]
node_Y = node[1::3]
node_Z = node[2::3]

# All (tetrahedral) elements:
elem = elemNodeTags[0].reshape(-1, 4)
elem = elem-1  # indexing from 0

# All (triangular) faces on the boundary (including subdomains):
face = faceNodeTags[0].reshape(-1, 3)
face = face-1  # indexing from 0

# Subdomain tags for each element:
n_elem = elem.shape[0]
min_elemTag = min(elemTags[0])
material = np.zeros((n_elem,), dtype=int)
for i in range(100):
    try:
        _, tags, _ = gmsh.model.mesh.getElements(dim=3, tag=i+1)
        material[tags[0]-min_elemTag] = i+1
    except:
        break

# Boundary tags for each face:
n_face = face.shape[0]
min_faceTag = min(faceTags[0])
boundary = np.zeros((n_face,), dtype=int)
for i in range(100):
    try:
        _, tags, _ = gmsh.model.mesh.getElements(dim=2, tag=i+1)
        boundary[tags[0]-min_faceTag] = i+1
    except:
        break


# Finalize the Gmsh API
gmsh.finalize()


N_show = 1000  # number of faces to show
if n_elem<N_show:
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(node_X, node_Y, node_Z, triangles=face, linewidths=0.2, edgecolor='k', color=[0, 1.0, 0.5, 0.1])
    ax.set_aspect('equal')
    ax.set_title("All faces on the boundary (including subdomains)")


# renumber boundary, set inner boundaries to zero
boundary_renumber = np.zeros(boundary.shape, dtype=np.int32)
k = 1
for i in [1, 2, 3, 4, 5, 7, 8, 9, 10, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27]:
    boundary_renumber[boundary == i] = k
    k += 1

# renumber boundary, set inner boundaries to zero
boundary_outer = np.zeros(boundary.shape, dtype=np.int32)
face_X = np.mean(node_X[face], axis=1)
face_Y = np.mean(node_Y[face], axis=1)
face_Z = np.mean(node_Z[face], axis=1)
boundary_outer[face_X==0] = 1
boundary_outer[face_X==x_sum] = 2
boundary_outer[face_Y==0] = 3
boundary_outer[face_Y==y_sum] = 4
boundary_outer[face_Z==0] = 5
boundary_outer[face_Z==z_sum] = 6

# show 3d scatter plot boundary faces, distinguish the surfaces by color:
if n_elem<N_show:
    plt.figure()
    ax = plt.axes(projection='3d')
    center_X = np.mean(node_X[face], axis=1)
    center_Y = np.mean(node_Y[face], axis=1)
    center_Z = np.mean(node_Z[face], axis=1)
    for i in [1, 2, 3, 4, 5, 6]:  # np.unique(boundary):
        color = [np.random.rand(), np.random.rand(), np.random.rand(), 0.3]
        ax.plot_trisurf(node_X, node_Y, node_Z, triangles=face[boundary_outer == i], linewidths=0.2, edgecolor='k', color=color)
    ax.set_aspect('equal')
    plt.show()

mesh = mesh_tools.Mesh3d(node_X, node_Y, node_Z, elem, face)
P2 = True  # use P2 elements
if P2:
    mesh.P1_to_PN(N=2)
    node_PN_X_mirror = z_sum/2-mesh.node_X
    node_PN = np.concatenate([node_PN_X_mirror.reshape(-1,1), mesh.node_Y.reshape(-1,1), mesh.node_Z.reshape(-1,1)], axis=1)
    elem_PN = np.concatenate([mesh.elem_PN[:,:7],mesh.elem_PN[:,8:],mesh.elem_PN[:,7].reshape(-1,1)], axis=1)
else:
    node_PN = np.empty((0,0))
    elem_PN = elem

np.unique(material)

# create "node" array with columns node_X, node_Y, node_Z
node_X_mirror = x_sum-node_X
node = np.concatenate([node_X_mirror.reshape(-1,1), node_Y.reshape(-1,1), node_Z.reshape(-1,1)], axis=1)
print("number of P1 nodes:", node.shape)
print("number of elements:", elem.shape)

name_all = ['nodeP1', 'node', 'elem', 'material', 'faceP1', 'face', 'boundary']
data_all = [node, node_PN, elem_PN, material, face, mesh.face_PN, boundary_outer]

compression='gzip'
compression_opts=9


p_iden = 'P2' if P2 else 'P1'

file_name = 'test_output.h5'

# Save the arrays to an HDF5 file
with h5py.File(file_name, 'w') as f:
    for name, data in zip(name_all, data_all):
        f.create_dataset(name=name, data=data, compression=compression, compression_opts=compression_opts)
