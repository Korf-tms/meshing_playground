import gmsh
import meshio
from numpy import sin, cos, pi, sqrt
from itertools import product
import os
from pathlib import Path


class Ellipse:
    """
    Ellipse as x**2/a**2 + y**2/b**2 = 1
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        if a > b:
            self.major_point_coo = (a, 0, 0)
        else:
            self.major_point_coo = (0, b, 0)

    def __lt__(self, other):
        """
        Presumes that the ellipses have no intersections and same major axes!
        """
        return self.a < other.a


class Ray:
    """
    Ray from origin given by angle from half-axis x > 0.
    """
    def __init__(self, phi):
        self.phi = phi

    def __eq__(self, other):
        return self.phi == other.phi

    def __lt__(self, other):
        return self.phi < other.phi

    def __str__(self):
        return f'Ray from origin, angle: {self.phi}'

    def __repr__(self):
        return repr(self.phi)


def ray_and_ellipse_intersection(ray, ellipse):
    """
    Compute the single intersection of a ray and an ellipse.
    :param ray:
    :param ellipse:
    :return: intersection_point as tupple
    """
    # easy peasy on computation on paper
    t = ellipse.a*ellipse.b/sqrt(ellipse.b**2*cos(ray.phi)**2 + ellipse.a**2*sin(ray.phi)**2)
    p_x = t*cos(ray.phi)
    p_y = t*sin(ray.phi)
    p_z = 0.0
    return p_x, p_y, p_z


def compute_region_data(ellipses_list, rays_list):
    """
    Compute points that defines the borders between regions.
    This pressumes aditional metainformation about the setup, suitable for out numerical experiments.
    :param ellipses_list: list of ellipses, each ellipse given by halfaxes, center is in 0,0, x and y axis
    :param rays_list: list of rays, given by angle taken from x > 0 axis
    :return: list of tuples of 4-points that define the regions and tuple with points on major axis
    """
    ellipses_list.sort()
    rays_list.sort()

    borders = []
    for closer_ellipse, farther_ellipse in zip(ellipses_list, ellipses_list[1:]):
        major_axis_smaller = closer_ellipse.major_point_coo
        major_axis_bigger = farther_ellipse.major_point_coo
        for i in range(len(rays_list)):
            ray1, ray2 = rays_list[i], rays_list[(i+1) % len(rays_list)]
            border_points = []
            for ray, ellipse in product((ray1, ray2), (closer_ellipse, farther_ellipse)):
                border_points.append(ray_and_ellipse_intersection(ray, ellipse))
            borders.append((border_points, (major_axis_smaller, major_axis_bigger)))
    # border points are ordered in the way what first two and second two are on the same ray
    # they are norm-wise ordered in the pairs, first smaller then bigger, in the same way as points on major axes
    return borders


def add_innermost_ellipse(model, ellipse, resolution, center_point):
    """
    Function to add ellipse to a model. Will be used to add the hole to the box mesh.
    :param model: (py)gmsh model
    :param ellipse: ellipse as instance of Ellipse class
    :param resolution: resolution of the ellipse points
    :param center_point: center point of the ellipse, as a point already in the model
    """
    a = ellipse.a
    b = ellipse.b
    points = [(a, 0, 0), (0, b, 0), (-a, 0, 0), (0, -b, 0)]
    points = [model.add_point(*point, meshSize=resolution) for point in points]
    if a > b:
        major_axis_index = 0
    else:
        major_axis_index = 1
    el_arcs = [model.add_ellipse_arc(startTag=points[i], centerTag=center_point, endTag=points[i+1],
                                     majorTag=points[major_axis_index]) for i in range(-1, 3)]
    el = model.add_curve_loop(el_arcs)
    return el


def create_mesh(rays_list, ellipses_list, corner_points, filename=None):
    """
    Creates square mesh with subdomains determined by rays and ellipses
    """
    ellipses_list.sort()

    resolution_outer = 10
    resolution_inner = 0.3

    # geometry init
    gmsh.initialize()
    model = gmsh.model.occ

    # center point needed for ellipses
    center_point = model.add_point(*(0, 0, 0))

    # construction of the outer box and region between outer border and inner ellipse
    border_points = [model.add_point(*point, meshSize=resolution_outer) for point in corner_points]
    lines_outer = [model.add_line(border_points[i], border_points[i+1]) for i in range(-1, len(corner_points)-1)]
    outer_loop = model.add_curve_loop(lines_outer)
    # biggest ellipse is used as border, that is why the ellipses list was sorted
    inner_loop = add_innermost_ellipse(model, ellipses_list[-1], resolution_inner, center_point)

    plane_surface = model.add_plane_surface((outer_loop, inner_loop))
    # model.add_physical(plane_surface, "Outer") TODO
    # gmsh.model.add_physical_group()

    # adds distinct regions, gradually filling the hole
    borders = compute_region_data(ellipses_list, rays_list)
    counter = 0
    for border, axis_coo in borders:
        # points to define the lines and ellipses
        aux_points = [model.add_point(*p, meshSize=resolution_inner) for p in border]
        # points on major axis needed by the engine to construct ellipse arcs
        axis_coo_p = [model.add_point(*p, meshSize=resolution_inner) for p in axis_coo]
        # arcs and lines defining the border
        arc1 = model.add_ellipse_arc(startTag=aux_points[2], endTag=aux_points[0], centerTag=center_point,
                                     majorTag=axis_coo_p[0])
        l0 = model.add_line(aux_points[0], aux_points[1])
        arc2 = model.add_ellipse_arc(startTag=aux_points[1], endTag=aux_points[3], centerTag=center_point,
                                     majorTag=axis_coo_p[1])
        l1 = model.add_line(aux_points[3], aux_points[2])
        cc = model.add_curve_loop([arc1, l0, arc2, l1])
        region = model.add_plane_surface((cc,))
        # model.add_physical(region, f'Region {counter}') TODO
        counter += 1

    model.synchronize()
    model.removeAllDuplicates()

    gmsh.model.mesh.generate(dim=2)
    gmsh.model.mesh.removeDuplicateNodes()
    if filename is not None:
        gmsh.write(filename)
    else:
        current_directory = os.getcwd()
        path = Path(current_directory) / 'mesh_output'
        if not path.exists():
            path.mkdir()
        gmsh.write('mesh_output/test_out.msh')


def create_xdmf_mesh_from_msh_file(filename):
    """
    Function converts triangle mesh from msh to fenicx prefered xdmf. Assumes that the mesh is 2D
    and prunes the extra z-coordinate.
    Assumes that the mesh has no physical groups! TODO: fix this
    """
    mesh_from_file = meshio.read(f"{filename}.msh")
    cells = mesh_from_file.get_cells_type("triangle")
    points = mesh_from_file.points[:, :2]  # removes the z-coordinate
    out_mesh = meshio.Mesh(points=points, cells={"triangle": cells})
    meshio.write(f"{filename}.xdmf", out_mesh)
    print(f"Mesh written to files: {filename}.xdmf, {filename}.h5")


if __name__ == "__main__":
    no_of_rays = 4
    rays = [Ray(2.0*pi*n/no_of_rays) for n in range(no_of_rays)]
    ellipses = [Ellipse(2.0*k, 3.0*k) for k in (1, 2, 4)]
    corners = [(50, -50, 0), (50, 50, 0), (-50, 50, 0), (-50, -50, 0)]

    create_mesh(rays, ellipses, corners)
    create_xdmf_mesh_from_msh_file('mesh_output/test_out')
