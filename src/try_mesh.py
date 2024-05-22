import dolfinx as dfx
from mpi4py import MPI


if __name__ == '__main__':
    import pyvista
    import numpy as np
    from dolfinx import plot

    # load mesh and domains data
    with dfx.io.XDMFFile(MPI.COMM_WORLD, 'mesh_output/test_out.xdmf', 'r') as mesh_file:
        mesh = mesh_file.read_mesh(name='Grid')
        cell_tags = mesh_file.read_meshtags(mesh, name='Grid')

    num_of_domains = len(set(cell_tags.values))
    # starting point for the 2D domain tags, given by gmsh mesh construction,
    # could be != 1 for nD meshes with physical groups of dim < n. TODO: explore
    marker_start = 1

    # generate random data for plot
    values = np.random.rand(num_of_domains)*100

    # create DG function and fill with numbers
    Q = dfx.fem.functionspace(mesh, ('DG', 0))
    k = dfx.fem.Function(Q)

    for marker, value in enumerate(values, start=marker_start):
        marked_cells = cell_tags.find(marker)
        k.x.array[marked_cells] = np.full_like(marked_cells, value)

    # plot created function using pyvista
    # https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals_code.html
    topology, cell_types, geometry = plot.vtk_mesh(mesh, 2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.cell_data["k"] = k.x.array.real
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    plotter.show()
