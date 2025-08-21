import gmsh
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import h5py


@dataclass
class PhysicalGroup:
    name: str = ''
    dim: int = -1
    group_tag: int = -1
    tags: list[int] = field(default_factory=list)


def _collect_tets_and_labels_from_phys():
    """Return (T, tet_labels). T is shape (ntet, 4 or 10). Labels align with rows."""
    conns, labels = [], []
    # Map 3D entity -> its physical tags (usually one)
    ent_to_phys = defaultdict(list)
    for d, ptag in gmsh.model.getPhysicalGroups(3):
        for ent in gmsh.model.getEntitiesForPhysicalGroup(3, ptag):
            ent_to_phys[ent].append(ptag)

    for ent, ptags in ent_to_phys.items():
        types, elemTags, elemNodes = gmsh.model.mesh.getElements(3, ent)
        for et, etags, enodes in zip(types, elemTags, elemNodes):
            name, _, _, nper, *_ = gmsh.model.mesh.getElementProperties(et)
            if not name.lower().startswith("tetra") or not len(etags):
                continue
            arr = np.array(enodes, dtype=np.int64).reshape(-1, nper)
            conns.append(arr)
            labels.append(np.full(arr.shape[0], ptags[0], dtype=np.int64))
    if conns:
        T = np.vstack(conns)
        tet_labels = np.concatenate(labels)
        assert tet_labels.shape[0] == T.shape[0]
        return T, tet_labels
    return np.empty((0, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)


def _collect_triangles_per_phys_surface():
    """Return (TRI, TRI_labels) by iterating each 2D physical surface entity
    and grabbing its triangle elements exactly as Gmsh produced them."""
    tris, labs = [], []
    for d, ptag in gmsh.model.getPhysicalGroups(2):
        ents = gmsh.model.getEntitiesForPhysicalGroup(2, ptag)
        for ent in ents:
            types, elemTags, elemNodes = gmsh.model.mesh.getElements(2, ent)
            for et, etags, enodes in zip(types, elemTags, elemNodes):
                name, _, _, nper, *_ = gmsh.model.mesh.getElementProperties(et)
                if not name.lower().startswith("triangle") or not len(etags):
                    continue
                arr = np.array(enodes, dtype=np.int64).reshape(-1, nper)
                tris.append(arr)
                labs.append(np.full(arr.shape[0], ptag, dtype=np.int64))
    if tris:
        return np.vstack(tris), np.concatenate(labs)
    return np.empty((0, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)


def write_hdf5_surfaces_from_phys_entities(out_path='slope_with_waterlevels.h5'):
    """
    Definitive exporter:
      • Tetrahedra + labels come from 3D physical volumes.
      • Surface triangles + labels come from 2D physical surface entities (exact as meshed).
      • We then compact to the tet-node set → NO face-only/orphan nodes.
      • Node coordinates use a tag→position map (correct after RCMK).
    """
    # Nodes & coordinates (post-RCMK)
    nodeTags, nodeXYZ, _ = gmsh.model.mesh.getNodes()
    nodeTags = nodeTags.astype(np.int64)
    coords = np.asarray(nodeXYZ, dtype=float).reshape(-1, 3)
    tag2pos = {int(t): i for i, t in enumerate(nodeTags)}  # tag -> row index in coords

    # Tetrahedra and labels
    T, tet_labels = _collect_tets_and_labels_from_phys()
    if T.size == 0:
        raise RuntimeError("No tetrahedra found in 3D physical groups.")
    tet_nper = T.shape[1]

    # Triangles and labels straight from 2D physicals
    TRI_raw, TRI_labels = _collect_triangles_per_phys_surface()

    # Keep only triangles fully on tet nodes (prevents reintroducing face-only nodes)
    tet_nodes_used = np.unique(T.ravel())
    if TRI_raw.size:
        mask = np.isin(TRI_raw, tet_nodes_used).all(axis=1)
        TRI = TRI_raw[mask]
        TRI_labels = TRI_labels[mask] if TRI_labels.size else TRI_labels
        # (Optional) warn if anything dropped:
        dropped = int(TRI_raw.shape[0] - TRI.shape[0])
        if dropped:
            print(f"Note: dropped {dropped} surface triangles not fully on tet nodes.")
    else:
        TRI = np.empty((0, 0), dtype=np.int64)
        TRI_labels = np.empty((0,), dtype=np.int64)

    # Compact to tet-only nodes, preserving order with tag→pos map
    tet_nodes_sorted = np.array(sorted(tet_nodes_used.tolist()), dtype=np.int64)
    tag_to_new = {int(t): i for i, t in enumerate(tet_nodes_sorted)}

    # points array
    points = np.empty((tet_nodes_sorted.size, 3), dtype=float)
    for t, i_new in tag_to_new.items():
        points[i_new] = coords[tag2pos[t]]

    # remap connectivity
    def remap(conn):
        if conn.size == 0:
            return conn
        out = np.empty_like(conn)
        it = np.nditer(conn, flags=['multi_index'])
        while not it.finished:
            out[it.multi_index] = tag_to_new[int(it[0])]
            it.iternext()
        return out

    tets_out = remap(T)
    tris_out = remap(TRI)

    # Write HDF5
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('points', data=points, compression='gzip', compression_opts=9)
        f.create_dataset('tetra_cells', data=tets_out, compression='gzip', compression_opts=9)
        if tet_labels.size:
            f.create_dataset('tetra_labels', data=tet_labels, compression='gzip', compression_opts=9)
        if tris_out.size:
            f.create_dataset('triangles', data=tris_out, compression='gzip', compression_opts=9)
            if TRI_labels.size:
                f.create_dataset('triangle_labels', data=TRI_labels, compression='gzip', compression_opts=9)

    # Small bandwidth proxy (on tets)
    span = int((tets_out.max(axis=1) - tets_out.min(axis=1)).max())
    print(f"HDF5 written: {out_path} | points={points.shape[0]} (tet-only) | "
          f"tets={tets_out.shape[0]} | tris={tris_out.shape[0]} | span≈{span}")

def create_mesh(file_name='slope_with_waterlevels',
                z_water_height=35, z_solid_water_level=50,
                h=20,
                order=2):
    """
    Generates a conforming mesh for a sloped geometry with specific boundaries
    cut at different water levels. This version focuses only on generating the
    correct geometry and mesh, without assigning physical groups.
    """
    def box_from_coordinates(coordinates):
        """Helper function to generate a box from given coordinates of its vertices."""
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

    gmsh.initialize()
    factory = gmsh.model.occ

    # --- GEOMETRY PARAMETERS ---
    x_slope = 60
    x_top = 85
    x_down = 30
    x_cover = 5
    z_layer0 = 20
    z_layer1 = 10
    z_layer2 = 30
    slope_half_width = 50
    turning_corner = 120

    z_sum = z_layer0 + z_layer1 + z_layer2
    concave = False
    angle = (360-turning_corner)/2/180*np.pi
    if turning_corner < 180:
        concave = True
        angle = turning_corner/2/180*np.pi
    y_half = slope_half_width * np.sin(angle)
    y_sum = 2*y_half
    x_cutout = slope_half_width * np.cos(angle)
    x_sum = x_top + x_slope + x_cutout + x_cover + x_down
    print(f"Model Dimensions: x={x_sum}, y={y_sum}, z={z_sum}")

    # --- INITIAL MESH CREATION ---
    bottom_layer = factory.add_box(0, 0, 0, x_sum, y_sum, z_layer0)
    middle_layer = factory.add_box(0, 0, z_layer0, x_sum, y_sum, z_layer1)
    top_layer = factory.add_box(0, 0, z_layer0+z_layer1, x_sum, y_sum, z_layer2)

    coordinates_cutout_left = [[x_sum - x_down - x_cover, 0, z_layer0+z_layer1], [x_sum, 0, z_layer0+z_layer1],
                            [x_sum, y_half, z_layer0+z_layer1], [x_sum - x_down - x_cover - x_cutout, y_half, z_layer0+z_layer1],
                            [x_top + x_cutout, 0, z_sum], [x_sum, 0, z_sum],
                            [x_sum, y_half, z_sum], [x_top, y_half, z_sum]]
    cutout_left = box_from_coordinates(coordinates_cutout_left)
    cutout_right_with_tag = factory.copy([(3, cutout_left)])
    factory.mirror(cutout_right_with_tag, 0, 1, 0, -y_half)

    coordinates_cover_left = [[x_sum - x_down - x_cover, 0, z_layer0+z_layer1], [x_sum - x_down, 0, z_layer0+z_layer1],
                            [x_sum - x_down - x_cutout, y_half, z_layer0+z_layer1], [x_sum - x_down - x_cover - x_cutout, y_half, z_layer0+z_layer1],
                            [x_top + x_cutout, 0, z_sum], [x_top + x_cutout + x_cover, 0, z_sum],
                            [x_top + x_cover, y_half, z_sum], [x_top, y_half, z_sum]]
    cover_left = box_from_coordinates(coordinates_cover_left)
    cover_right_with_tag = factory.copy([(3, cover_left)])
    cover_right = cover_right_with_tag[0][1]
    factory.mirror(cover_right_with_tag, 0, 1, 0, -y_half)

    if concave:
        factory.translate([(3, cutout_left)], 0, y_half, 0)
        factory.translate(cutout_right_with_tag, 0, -y_half, 0)
        factory.translate([(3, cover_left)], 0, y_half, 0)
        factory.translate(cover_right_with_tag, 0, -y_half, 0)

    cut_output = factory.cut([(3, top_layer)], [(3, cutout_left)]+cutout_right_with_tag)
    top_layer_final = cut_output[0][0][1]
    factory.synchronize()

    # --- CUT THE COVER VOLUMES ---
    eps = 10
    p1 = factory.add_point(0-eps, 0-eps, z_water_height); p2 = factory.add_point(x_sum+eps, 0-eps, z_water_height)
    p3 = factory.add_point(x_sum+eps, y_sum+eps, z_water_height); p4 = factory.add_point(0-eps, y_sum+eps, z_water_height)
    cl_water = factory.add_curve_loop([factory.add_line(p1,p2), factory.add_line(p2,p3), factory.add_line(p3,p4), factory.add_line(p4,p1)])
    cutting_plane_water = factory.add_plane_surface([cl_water])
    
    covers_to_cut = [(3, cover_left), (3, cover_right)]
    out_tags_front, cut_map_covers = factory.fragment(covers_to_cut, [(2, cutting_plane_water)], removeTool=True)
    factory.synchronize()
    # Remove the cutting plane surface from the output tags.
    for dimTag in out_tags_front:
        if dimTag[0] == 2:
            factory.remove([dimTag], recursive=True)
            out_tags_front.remove(dimTag)
            factory.synchronize()
            break
    factory.synchronize()

    # --- CUT THE TOP LAYER VOLUME ---
    p5 = factory.add_point(0-eps, 0-eps, z_solid_water_level); p6 = factory.add_point(x_sum+eps, 0-eps, z_solid_water_level)
    p7 = factory.add_point(x_sum+eps, y_sum+eps, z_solid_water_level); p8 = factory.add_point(0-eps, y_sum+eps, z_solid_water_level)
    cl_solid = factory.add_curve_loop([factory.add_line(p5,p6), factory.add_line(p6,p7), factory.add_line(p7,p8), factory.add_line(p8,p5)])
    cutting_plane_solid = factory.add_plane_surface([cl_solid])

    out_tags_top, cut_map_top = factory.fragment([(3, top_layer_final)], [(2, cutting_plane_solid)], removeTool=True)
    # Remove the cutting plane surface from the output tags.
    for dimTag in out_tags_top:
        if dimTag[0] == 2:
            factory.remove([dimTag], recursive=True)
            out_tags_top.remove(dimTag)
            factory.synchronize()
            break
    factory.synchronize()
    
    # --- 6. FINAL GLOBAL STITCHING ---
    # This step takes all the final volume parts (some pre-cut, some untouched)
    # and makes all their shared boundaries conformal.
    final_assembly_list = [ (3, bottom_layer), (3, middle_layer)] + out_tags_top + out_tags_front
    factory.fragment(final_assembly_list, [])
    factory.synchronize()

    # --- PHYSICAL GROUP ASSIGNMENT ---
    names3d = ['bottom_layer', 'middle_layer', 'top_layer', 'cover']
    names2d = ['dry_slope', 'wet_slope', 'dry_solid', 'wet_solid', 'x_max', 'y0', 'y_max', 'z0', 'z_max', 'z_water_bed']
    groups = {name: PhysicalGroup(name=name, dim=3, group_tag=i) for i, name in enumerate(names3d, start=1)} | \
        {name: PhysicalGroup(name=name, dim=2, group_tag=i) for i, name in enumerate(names2d, start=len(names3d)+1)}
    groups['bottom_layer'].tags = [bottom_layer]
    groups['middle_layer'].tags = [middle_layer]
    groups['top_layer'].tags = [tag for _, tag in out_tags_top]
    groups['cover'].tags = [tag for _, tag in out_tags_front]

    # faces without cuts
    positions = {  # name: (axis, coordinate value)
        'x_max': (0, x_sum),
        'y0': (1, 0),
        'y_max': (1, y_sum),
        'z0': (2, 0),
        'z_max': (2, z_sum),
        'z_water_bed': (2, z_layer0+z_layer1)
    }
    all_faces = factory.get_entities(dim=2)
    # very geometry specfic and brittle
    for dim, tag in all_faces:
        center_coo = factory.get_center_of_mass(dim, tag)
        for name, (axis, value) in positions.items():
            if np.isclose(center_coo[axis], value, atol=1e-5):
                if name == 'z_water_bed' and center_coo[0] < x_sum - x_down - x_cover:
                    continue
                groups[name].tags.append(tag)
                break
            # faces at x = 0
            if np.isclose(center_coo[0], 0, atol=1e-5):
                if center_coo[2] < z_solid_water_level:
                    groups['wet_solid'].tags.append(tag)
                elif center_coo[2] > z_solid_water_level:
                    groups['dry_solid'].tags.append(tag)
                else:
                    raise ValueError(f"Unexpected face at x=0 with center at z={center_coo[2]}")
    
    # faces at the slope
    for dim, tag in out_tags_front:
        center_coo = factory.get_center_of_mass(dim, tag)
        boundary_face = find_outer_slope_face(tag)
        if center_coo[2] > z_water_height:
            groups['dry_slope'].tags.append(boundary_face)
        else:
            groups['wet_slope'].tags.append(boundary_face)

    # add physical groups to the model    
    for _, group in groups.items():
        gmsh.model.addPhysicalGroup(group.dim, group.tags, tag=group.group_tag, name=group.name)

    # --- 8. MESHING ---
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(order)
    gmsh.model.mesh.remove_duplicate_elements()
    gmsh.model.mesh.remove_duplicate_nodes()

    oldTags, newTags = gmsh.model.mesh.computeRenumbering(method="RCMK")
    gmsh.model.mesh.renumberNodes(oldTags, newTags)
    gmsh.model.mesh.renumberElements()

    # gmsh.write(f"{file_name}.msh")
    # print(f"Mesh written to {file_name}.msh")
    
    # Launch the GUI to inspect the final mesh
    # gmsh.fltk.run()

    write_hdf5_surfaces_from_phys_entities(f"{file_name}.h5")

    gmsh.finalize()

def transform_to_hdf5(input_file='slope_with_waterlevels.msh'):
    import meshio
    import h5py


    mesh = meshio.read(input_file)
    points = mesh.points

    # element order for P2 tetras: A, B, C, D, AB, BC, CA, AD, DB, DC
    # element order for P2 triangles: A, B, C, AB, BC, CA

    with h5py.File(input_file.replace('.msh', '.h5'), 'w') as f:
        f.create_dataset('points', data=points, compression='gzip', compression_opts=9)
        for name, cells in mesh.cells_dict.items():
            if name.startswith('tetra'):
                f.create_dataset('tetra_cells', data=cells, compression='gzip', compression_opts=9)
                f.create_dataset('tetra_labels', data=mesh.get_cell_data('gmsh:physical', name),
                                 compression='gzip', compression_opts=9)
            elif name.startswith('triangle'):
                f.create_dataset('triangles', data=cells, compression='gzip', compression_opts=9)
                f.create_dataset('triangle_labels', data=mesh.get_cell_data('gmsh:physical', name),
                                 compression='gzip', compression_opts=9)

    print(f"Transformed mesh saved to {input_file.replace('.msh', '.h5')}")

if __name__ == "__main__":
    create_mesh(h=15)
    # transform_to_hdf5()