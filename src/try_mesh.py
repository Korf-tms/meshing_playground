import dolfinx as dfx
from mpi4py import MPI
from ellipses_regions_generator import generate_tsx_mesh_with_regions


if __name__ == '__main__':
    import pyvista
    import numpy as np
    from dolfinx import plot
    import ufl
    from dolfinx.fem.petsc import LinearProblem

    generate_tsx_mesh_with_regions()

    show_pictures = True

    # load mesh and domains data
    # https://github.com/jorgensd/dolfinx-tutorial/blob/v0.8.0/chapter3/subdomains.ipynb
    with dfx.io.XDMFFile(MPI.COMM_WORLD, 'tsx_ellipses_regions.xdmf', 'r') as mesh_file:
        mesh = mesh_file.read_mesh(name='Grid')
        cell_tags = mesh_file.read_meshtags(mesh, name='Grid')

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)  # taken from the link, why is it here?

    # prepare tags for boudary conditions
    with dfx.io.XDMFFile(MPI.COMM_WORLD, 'tsx_ellipses_regions_boundary.xdmf', 'r') as boundary_file:
        facet_tags = boundary_file.read_meshtags(mesh, name='Grid')

    num_of_domains = len(set(cell_tags.values))

    # starting point for the 2D domain tags, given by gmsh mesh construction,
    # could be != 1 for nD meshes with physical groups of dim < n. TODO: explore
    marker_start = 1  # effectively min(set(cell_tags.values)), most of the time

    # generate random data for coefficient function
    np.random.seed(25)
    values = np.random.rand(num_of_domains)*1e3

    # create DG function and fill with data
    Q = dfx.fem.functionspace(mesh, ('DG', 0))
    k = dfx.fem.Function(Q)

    for marker, value in enumerate(values, start=marker_start):
        marked_cells = cell_tags.find(marker)
        k.x.array[marked_cells] = np.full_like(marked_cells, value)

    if show_pictures:
        # plot created coefficient function using pyvista
        # https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals_code.html
        topology, cell_types, geometry = plot.vtk_mesh(mesh, 2)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        grid.cell_data["k"] = k.x.array.real
        plotter = pyvista.Plotter()
        plotter.title = 'Coefficients'
        plotter.add_mesh(grid, show_edges=True, scalar_bar_args={'vertical': True})
        plotter.view_xy()
        plotter.show()

    # solve a little test problem to test the coefficients and boundaries
    V = dfx.fem.functionspace(mesh, ('Lagrange', 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(k * ufl.grad(u), ufl.grad(v))*ufl.dx
    L = dfx.fem.Constant(mesh, dfx.default_scalar_type(2.0e2))*v*ufl.dx

    # set bcs
    outer_facets = facet_tags.find(18)  # magical number given by the mesh construction, =max(facet_tags.values)
    inner_facets = facet_tags.find(17)  # magical number given by the mesh construction, =min(facet_tags.values)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)  # taken from the link, why is it here?
    inner_facet_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, inner_facets)
    outer_facets_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, outer_facets)
    bcs = [dfx.fem.dirichletbc(dfx.default_scalar_type(1.0e1), outer_facets_dofs, V),
           dfx.fem.dirichletbc(dfx.default_scalar_type(1.0e2), inner_facet_dofs, V)]

    problem = LinearProblem(a, L, bcs=bcs, petsc_options={'ksp_type: preonly,'
                                                          'pc_type': 'lu'})
    uh = problem.solve()

    if show_pictures:
        grid_uh = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
        grid_uh.point_data["u"] = uh.x.array.real
        grid_uh.set_active_scalars("u")
        p2 = pyvista.Plotter()
        p2.title = 'Solution'
        p2.add_mesh(grid_uh, show_edges=True, scalar_bar_args={'vertical': True})
        p2.show()
        warped = grid_uh.warp_by_scalar()
        p3 = pyvista.Plotter()
        p3.title = 'Solution in 3d'
        p3.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalar_bar_args={'vertical': True})
        p3.show()




