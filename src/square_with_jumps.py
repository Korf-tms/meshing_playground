import numpy as np

import dolfinx as dfx
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from ufl import grad, div, inner, dx
from dolfinx.fem import Constant
from basix.ufl import element, mixed_element
from dolfinx.fem.petsc import LinearProblem


# TODO: move to separate file
def load_mesh_and_domain_tags(path_to_mesh):
    """
    Taken from:
    https://github.com/jorgensd/dolfinx-tutorial/blob/v0.8.0/chapter3/subdomains.ipynb
    """
    with dfx.io.XDMFFile(MPI.COMM_WORLD, f'{path_to_mesh}.xdmf', 'r') as mesh_file:
        mesh = mesh_file.read_mesh(name='Grid')
        cell_tags = mesh_file.read_meshtags(mesh, name='Grid')

    # taken from the link, no idea what it does
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    with dfx.io.XDMFFile(MPI.COMM_WORLD, f'{path_to_mesh}_boundary.xdmf', 'r') as boundary_file:
        facet_tags = boundary_file.read_meshtags(mesh, name='Grid')

    return mesh, cell_tags, facet_tags

# TODO: move to separate file
def prepare_coefficient_functions(mesh, cells_tags,
                                  lmbda_list, mu_list, alpha_list, cpp_list, k_list):
    # prepare space and functions
    Q = dfx.fem.functionspace(mesh, ('DG', 0))
    lmbda = dfx.fem.Function(Q)
    mu = dfx.fem.Function(Q)
    alpha = dfx.fem.Function(Q)
    cpp = dfx.fem.Function(Q)
    k = dfx.fem.Function(Q)

    list_of_lists = [lmbda_list, mu_list, alpha_list, cpp_list, k_list]
    list_of_functions = [lmbda, mu, alpha, cpp, k]

    # effectively min(set(cell_tags.values)), most of the time, depends on gmsh code
    marker_start = 1

    # create dict to store map of domains to actual cells
    tag_to_cells = {marker: cells_tags.find(marker) for marker in set(cells_tags.values)}

    for parameter_function, list_of_values in zip(list_of_functions, list_of_lists):
        for marker, value in enumerate(list_of_values, start=marker_start):
            cell_numbers = tag_to_cells[marker]
            parameter_function.x.array[cell_numbers] = value

        parameter_function.x.scatter_forward()

    return lmbda, mu, alpha, cpp, k


def epsilon(u):
    return ufl.sym(ufl.nabla_grad(u))


def setup_and_compute(mesh, facet_tags,
                      lmbda, mu, alpha, cpp, k, kmin,
                      tau_f, t_steps_num):

    tau = Constant(mesh, tau_f)
    P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    P1 = element("Lagrange", mesh.basix_cell(), 1)
    RT = element("Raviart-Thomas", mesh.basix_cell(), 1)  # TODO: check the degree convention
    V_element = mixed_element([P2, RT, P1])
    V = dfx.fem.functionspace(mesh, V_element)
    u, v, p = ufl.TrialFunctions(V)
    w, z, q = ufl.TestFunctions(V)
    x_h = dfx.fem.Function(V)  # solution of the current timestep

    # boundary conditions, hardcoded for this version, TODO: move to function, make into input
    top_facets_inner = facet_tags.find(101)
    top_facets_outer = facet_tags.find(102)
    top_facets = np.concatenate((top_facets_inner, top_facets_outer))
    left_facets = facet_tags.find(103)
    right_facets = facet_tags.find(104)
    bottom_facets = facet_tags.find(105)

    V1, _ = V.sub(1).collapse()
    left_v_bc_function = dfx.fem.Function(V1)
    right_v_bc_function = dfx.fem.Function(V1)

    # Assign values [0.0, 0.0] to the functions on the appropriate subspace
    left_v_bc_function.x.array[:] = 0.0
    right_v_bc_function.x.array[:] = 0.0

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    left_u_dofs = dfx.fem.locate_dofs_topological(V.sub(0).sub(0), 1, left_facets)
    right_u_dofs = dfx.fem.locate_dofs_topological(V.sub(0).sub(0), 1, right_facets)
    bottom_u_dofs = dfx.fem.locate_dofs_topological(V.sub(0).sub(1), 1, bottom_facets)
    left_v_dofs = dfx.fem.locate_dofs_topological((V.sub(1), V1), 1, left_facets)
    right_v_dofs = dfx.fem.locate_dofs_topological((V.sub(1), V1), 1, right_facets)

    dirichlet_bcs = [
        dfx.fem.dirichletbc(dfx.default_scalar_type(0.0), left_u_dofs, V.sub(0).sub(0)),
        dfx.fem.dirichletbc(dfx.default_scalar_type(0.0), right_u_dofs, V.sub(0).sub(0)),
        dfx.fem.dirichletbc(dfx.default_scalar_type(0.0), bottom_u_dofs, V.sub(0).sub(1)),
        dfx.fem.dirichletbc(left_v_bc_function, left_v_dofs, V1),
        dfx.fem.dirichletbc(right_v_bc_function, right_v_dofs, V1),
    ]

    # forms and preconditioners
    a_elastic = 2*mu*inner(epsilon(u), epsilon(w))*dx + lmbda*div(u)*div(w)*dx
    a = a_elastic + alpha*p*div(w)*dx + \
        tau/k*inner(v, z)*dx + tau*p*div(z)*dx + \
        alpha*div(u)*q*dx + tau*div(v)*q*dx - cpp*p*q*dx
    a = dfx.fem.form(a)

    divdiv_term = (cpp + alpha ** 2 / (lmbda + 2 * mu) + tau * k)
    pbnb = a_elastic + \
           tau * (1.0/ k) * inner(v, z) * dx + tau ** 2 / divdiv_term * div(v) * div(z) * dx + \
           divdiv_term * p * q * dx

    # rhs including contribution from boundary conditions
    ds_top_inner = ufl.Measure("ds", domain=mesh, subdomain_data=top_facets_inner)
    ds_top = ufl.Measure("ds", domain=mesh, subdomain_data=top_facets)

    f = Constant(mesh, np.array([0.0, 0.0]))
    g = Constant(mesh, np.array([0.0, 0.0]))
    h = Constant(mesh, dfx.default_scalar_type(0.0))

    n = ufl.FacetNormal(mesh)

    p_flux = 3e6
    p_stress = 4e6
    rhs = inner(f, w)*dx + inner(g, z)*dx + h*q*dx
    # rhs += p_stress*tau*inner(n, w)*ds_top_inner(0)
    # rhs += p_flux*inner(n, z)*ds_top(0)
    rhs_form = dfx.fem.form(rhs)

    A = dfx.fem.petsc.assemble_matrix(a, bcs=dirichlet_bcs)
    A.assemble()

    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A, A)
    solver.setType('preonly')
    solver.getPC().setType('lu')
    opts = PETSc.Options()
    opts['pc_factor_mat_solver_type'] = 'mumps'
    opts['ksp_converged_reason'] = True
    solver.setFromOptions()

    current_time = 0.0

    for step in range(0, t_steps_num):
        print(f'step: {step}')
        current_time += tau_f

        b = dfx.fem.petsc.assemble_vector(rhs_form)
        dfx.fem.petsc.apply_lifting(b, [a], [dirichlet_bcs])  # ???
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dfx.fem.set_bc(b, dirichlet_bcs)
        solver.solve(b, x_h.x.petsc_vec)



if __name__ == '__main__':
    SEC_IN_DAY = 24 * 60 * 60
    young_e = 6e10
    poisson_nu = 0.2
    mu = young_e / (2 * (1 + poisson_nu))
    lmbda = young_e * poisson_nu / ((1 + poisson_nu) * (1 - 2 * poisson_nu))
    alpha = 0.2
    cpp = 7.712e-12
    timestep = SEC_IN_DAY / 2

    square_mesh, cell_tags, facet_tags = load_mesh_and_domain_tags('square_mesh')

    # NOTE: following assumes that domains are label consecutively from 1, gmsh does this for now
    number_of_subdomains = max(set(cell_tags.values))
    lmbda_values = [lmbda] * number_of_subdomains
    mu_values = [mu] * number_of_subdomains
    alpha_values = [alpha] * number_of_subdomains
    cpp_values = [cpp] * number_of_subdomains
    k_values = [6.0e-19 + _*1.0e-20 for _ in range(number_of_subdomains)]  # TODO: lognormal
    kmin = min(k_values)

    lmbda, mu, alpha, cpp, k = prepare_coefficient_functions(square_mesh, cell_tags,
                                                              lmbda_values, mu_values, alpha_values, cpp_values, k_values)

    setup_and_compute(square_mesh, facet_tags, lmbda, mu, alpha, cpp, k, kmin, tau_f=timestep, t_steps_num=1)




