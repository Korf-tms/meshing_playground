import gmsh
import numpy as np


def create_mesh(file_name='slope_with_waterlevels',
                z_water_height=35, z_solid_water_level=50,
                h=3,
                order=2):
    def box_from_coordinates(coordinates):
        """
        Helper function to generate box from given coordinates of its vertices.
        """
        points = [factory.add_point(*c) for c in coordinates]
        lines = [factory.add_line(points[i], points[(i+1)%4]) for i in range(4)]
        lines += [factory.add_line(points[i+4], points[(i+1)%4+4]) for i in range(4)]
        lines += [factory.add_line(points[i], points[i+4]) for i in range(4)]
        surfaces = []
        loops = [[0,1,2,3],[4,5,6,7],[0,8,4,9],[1,9,5,10],[2,10,6,11],[3,11,7,8]]
        for loop_indices in loops:
            cl = factory.add_curve_loop([lines[i] for i in loop_indices])
            surfaces.append(factory.add_plane_surface([cl]))
        sl = factory.add_surface_loop(surfaces)
        return factory.add_volume([sl])

    def find_outer_slope_face(volume_tag):
        """
        Finds the single "outer" sloped face of a volume by checking two criteria:
        1. It must be an exterior face (adjacent to only one volume).
        2. It must be slanted in all 3 dimensions (its bounding box is not flat).
        """
        boundaries = gmsh.model.getBoundary([(3, volume_tag)], oriented=False)
        eps = 1e-6 # ok value for the slopes in meters

        for dim, tag in boundaries:
            adj_vols, _ = gmsh.model.getAdjacencies(dim, tag)
            if len(adj_vols) == 1:  # It's an exterior face. Now check the geometry.
                min_x, min_y, min_z, max_x, max_y, max_z = gmsh.model.getBoundingBox(dim, tag)

                # Criterion 2: Is it slanted in all 3 dimensions?
                is_sloped_in_x = (max_x - min_x) > eps
                is_sloped_in_y = (max_y - min_y) > eps
                is_sloped_in_z = (max_z - min_z) > eps

                if is_sloped_in_x and is_sloped_in_y and is_sloped_in_z:  # there is only one suitable face
                    return tag
        return -1 # Return -1 if not found, should actually raise some error here
    
    def intersect_with_z_level_plane(z_level, face_to_cut_id):
        eps = 10  # safe distance from the slope object mass
        p1 = factory.add_point(0-eps, 0-eps, z_level); p2 = factory.add_point(x_sum+eps, 0-eps, z_level)
        p3 = factory.add_point(x_sum+eps, y_sum+eps, z_level); p4 = factory.add_point(0-eps, y_sum+eps, z_level)
        cutting_plane_loop = factory.add_curve_loop([factory.add_line(p1,p2), factory.add_line(p2,p3), factory.add_line(p3,p4), factory.add_line(p4,p1)])
        cutting_plane_surface = factory.add_plane_surface([cutting_plane_loop])

        # Use fragment to split the faces
        out_tags_split, _ = factory.fragment(
            [(2, face_to_cut_id)], # Objects to split
            [(2, cutting_plane_surface)],  # Tool
        )
        # The output will contain the new upper and lower face parts, and the new intersection object.
        upper, lower = -1, -1
        for dimTag in out_tags_split:
            center_coo = factory.getCenterOfMass(*dimTag)
            if np.isclose(center_coo[2], z_level):  # intersection in the plane, we do not want that
                pass
            elif center_coo[2] > z_level:
                upper = dimTag[1]
            else:
                lower = dimTag[1]
        
        factory.remove([(2, cutting_plane_surface)], recursive=True)
        factory.synchronize()  # is this actually needed here?
        return upper, lower  # should probably raise some error if any is -1

    gmsh.initialize()
    factory = gmsh.model.occ

    # parameters of 3d geometry
    x_slope = 60  # 15, 35, 60
    x_top = 85
    x_down = 30
    x_cover = 5
    z_layer0 = 20
    z_layer1 = 10
    z_layer2 = 30
    slope_half_width = 50
    turning_corner = 120  # 90, 120, 135, 150, 270, 240, 225, 210

    z_sum = z_layer0 + z_layer1 + z_layer2
    concave = False
    angle = (360-turning_corner)/2/180*np.pi
    if turning_corner < 180:
        concave = True
        angle = turning_corner/2/180*np.pi
    y_half = slope_half_width * np.sin(angle)
    y_sum = 2*y_half
    x_cutout = slope_half_width * np.cos(angle)
    x_sum = x_top + x_slope + x_cutout + x_cover + x_down  # discrepancy in the paper, x_cutout added
    print(x_sum, y_sum, z_sum)


    bottom_layer = factory.add_box(0, 0, 0, x_sum, y_sum, z_layer0)
    middle_layer = factory.add_box(0, 0, z_layer0, x_sum, y_sum, z_layer1)
    top_layer = factory.add_box(0, 0, z_layer0+z_layer1, x_sum, y_sum, z_layer2)

    # prepare deformed box to cut out from the top layer (left side):
    coordinates_cutout_left = [[x_sum - x_down - x_cover, 0, z_layer0+z_layer1], [x_sum, 0, z_layer0+z_layer1],
                            [x_sum, y_half, z_layer0+z_layer1], [x_sum - x_down - x_cover - x_cutout, y_half, z_layer0+z_layer1],
                            [x_top + x_cutout, 0, z_sum], [x_sum, 0, z_sum],
                            [x_sum, y_half, z_sum], [x_top, y_half, z_sum]]
    cutout_left = box_from_coordinates(coordinates_cutout_left)

    # right side
    cutout_right_with_tag = factory.copy([(3, cutout_left)])
    factory.mirror(cutout_right_with_tag, 0, 1, 0, -y_half)

    # left side of slope cover:
    coordinates_cover_left = [[x_sum - x_down - x_cover, 0, z_layer0+z_layer1], [x_sum - x_down, 0, z_layer0+z_layer1],
                            [x_sum - x_down - x_cutout, y_half, z_layer0+z_layer1], [x_sum - x_down - x_cover - x_cutout, y_half, z_layer0+z_layer1],
                            [x_top + x_cutout, 0, z_sum], [x_top + x_cutout + x_cover, 0, z_sum],
                            [x_top + x_cover, y_half, z_sum], [x_top, y_half, z_sum]]
    cover_left = box_from_coordinates(coordinates_cover_left)

    # right side:
    cover_right_with_tag = factory.copy([(3, cover_left)])
    cover_right = cover_right_with_tag[0][1]
    factory.mirror(cover_right_with_tag, 0, 1, 0, -y_half)

    if concave:  # swap left and right side
        factory.translate([(3, cutout_left)], 0, y_half, 0)
        factory.translate(cutout_right_with_tag, 0, -y_half, 0)
        factory.translate([(3, cover_left)], 0, y_half, 0)
        factory.translate(cover_right_with_tag, 0, -y_half, 0)

    # cut out prepared deformed boxes:
    cut_output = factory.cut([(3, top_layer)], [(3, cutout_left)]+cutout_right_with_tag)
    top_layer_final = cut_output[0][0][1]

    # Puts the volumes together
    final_volumes_to_heal = [(3, bottom_layer), (3, middle_layer), (3, top_layer_final), (3, cover_left), (3, cover_right)]
    out_tags, out_map = factory.fragment(final_volumes_to_heal, [])
    factory.synchronize()
    # here, out map is the same as the input, everything retains its tag, so no action needed
    print(final_volumes_to_heal)
    print(out_map)

    # no renumbering needed here as well
    new_cover_left_tag = cover_left
    new_cover_right_tag = cover_right

    # Find the specific faces using the helper function
    slope_face_left_tag = find_outer_slope_face(new_cover_left_tag)
    slope_face_right_tag = find_outer_slope_face(new_cover_right_tag)

    if slope_face_left_tag == -1 or slope_face_right_tag == -1:  # should raise some errors here
        print('ERROR: Failed to find the outer slope surfaces on the healed model.')

    upper_l, lower_l = intersect_with_z_level_plane(z_level=z_water_height, face_to_cut_id=slope_face_left_tag)
    upper_r, lower_r = intersect_with_z_level_plane(z_level=z_water_height, face_to_cut_id=slope_face_right_tag)

    # deal with the cut at x = 0
    all_faces = gmsh.model.getEntities(dim=2)
    max_z_center = -1
    x0_faces = []  # will contain the wet x0 faces
    # we must pick the correct face to cut depending on z_solid_water_level!
    # currently provided by naked eye that the cut is in the topmost face
    for dim, tag in all_faces:
        center_coo = factory.get_center_of_mass(dim, tag)
        if abs(center_coo[0]) < 1e-6:  # face is at the x=0 plane
            x0_faces.append(tag)
            if max_z_center < center_coo[2]:
                max_z_center = center_coo[2]
                top_face = tag
    x0_faces.remove(top_face)  # top face will be cut into two, we add the resulting lower face

    upper_x0, lower_x0 = intersect_with_z_level_plane(z_level=z_solid_water_level, face_to_cut_id=top_face)
    x0_faces.append(lower_x0)

    positions = {  # name: (axis, coordinate value)
        'x_max': (0, x_sum),
        'y0': (1, 0),
        'ymax': (1, y_sum),
        'z0': (2, 0),
        'zmax': (2, z_sum),
        'zwater_bed': (2, z_layer0+z_layer1)
    }

    # mark the rest of the boundary faces
    face_groups = {}
    for dim, tag in all_faces:
        center_coo = factory.get_center_of_mass(dim, tag)
        for name, (axis, value) in positions.items():
            if np.isclose(center_coo[axis], value):
                if name not in face_groups:
                    face_groups[name] = []
                # special case for the water bed, we want only the boundary face
                # we know that obundary face is on the right from the end of the slope
                if name == 'zwater_bed' and center_coo[0] < x_sum - x_down - x_cover:
                    continue
                face_groups[name].append(tag)

    name2tag = {'bottom_layer': 1, 'middle_layer': 2, 'top_layer': 3, 'cover': 4,
                'dry_slope': 5, 'wet_slope': 6,
                'dry_solid': 7, 'wet_solid': 8,
                'x_max': 9, 'y0': 10, 'ymax': 11, 'z0': 12, 'zmax': 13, 'zwater_bed': 14}

    # Assign physical groups
    gmsh.model.addPhysicalGroup(3, [bottom_layer], name="bottom_layer", tag=name2tag['bottom_layer'])
    gmsh.model.addPhysicalGroup(3, [middle_layer], name="middle_layer", tag=name2tag['middle_layer'])
    gmsh.model.addPhysicalGroup(3, [top_layer_final], name="top_layer", tag=name2tag['top_layer'])
    gmsh.model.addPhysicalGroup(3, [new_cover_left_tag, new_cover_right_tag], name="cover", tag=name2tag['cover'])
    gmsh.model.addPhysicalGroup(2, [upper_l, upper_r], name='dry_slope', tag=name2tag['dry_slope'])
    gmsh.model.addPhysicalGroup(2, [lower_l, lower_r], name='wet_slope', tag=name2tag['wet_slope'])
    gmsh.model.addPhysicalGroup(2, [upper_x0], name='dry_solid', tag=name2tag['dry_solid'])
    gmsh.model.addPhysicalGroup(2, x0_faces, name='wet_solid', tag=name2tag['wet_solid'])
    for name, tags in face_groups.items():
        gmsh.model.addPhysicalGroup(2, tags, name=name, tag=name2tag[name])

    factory.synchronize()

    # set sizes and generate mesh, the background field should probably go somewhere here?
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)
    gmsh.write(f"{file_name}.msh")
    print(f"Mesh written to {file_name}")
    gmsh.fltk.run()
    gmsh.finalize()

    return name2tag


def transform_to_hdf5(input_file='slope_with_waterlevels.msh', order=2):
    import meshio
    import h5py
    from math import comb

    # exploiting gmsh naming convention
    triangle_num = '' if order==1 else comb(order+2, 2)
    tetra_num = '' if order==1 else comb(order+3, 3)

    mesh = meshio.read(input_file)
    points = mesh.points

    # element order for P2 tetras: A, B, C, D, AB, BC, CA, AD, DB, DC
    tetra_cells = mesh.get_cells_type(f"tetra{tetra_num}")
    # element order for P2 triangles: A, B, C, AB, BC, CA
    triangles = mesh.get_cells_type(f"triangle{triangle_num}")

    tetra_labels = mesh.get_cell_data("gmsh:physical", f"tetra{tetra_num}")
    triangle_labels = mesh.get_cell_data("gmsh:physical", f"triangle{triangle_num}")

    with h5py.File(input_file.replace('.msh', '.h5'), 'w') as f:
        f.create_dataset('points', data=points, compression='gzip', compression_opts=9)
        f.create_dataset('tetra_cells', data=tetra_cells.data, compression='gzip', compression_opts=9)
        f.create_dataset('triangles', data=triangles.data, compression='gzip', compression_opts=9)
        f.create_dataset('tetra_labels', data=tetra_labels, compression='gzip', compression_opts=9)
        f.create_dataset('triangle_labels', data=triangle_labels, compression='gzip', compression_opts=9)

    print(f"Transformed mesh saved to {input_file.replace('.msh', '.h5')}")

if __name__ == '__main__':
    description = create_mesh()
    for name, tag in description.items():
        print(f"{name}: {tag}")
    transform_to_hdf5()
