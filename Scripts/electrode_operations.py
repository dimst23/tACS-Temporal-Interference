import pymesh
import numpy as np

def orient_electrode(mesh, init_point):
    """Orient the electrode along the surface. Connectivy shall be enabled in the mesh that has been given.

    Args:
        mesh ([type]): [description]
        init_point ([type]): [description:

    Returns:
        [type]: [description]
    """
	point_id = np.where(np.sum(mesh.vertices == init_point, axis=1))[
            0][0]  # Unique point assumed
	face = mesh.get_vertex_adjacent_faces(point_id)[0]

	points = []
	for point in mesh.vertices[mesh.faces[face]]:
		if np.sum(point != init_point):
			points.append(point)
	p_1 = points[0] - init_point
	p_2 = points[1] - init_point

	normal = np.cross(p_1, p_2)
	return normal/np.linalg.norm(normal)

def add_electrode(surface_mesh, electrode_mesh):
	# Get the surface outline including the electrode
	model = pymesh.merge_meshes((surface_mesh, electrode_mesh))
	outer_hull = pymesh.compute_outer_hull(model)
	
	# Create the surface with the electrode mesh imprinted
	electrode_tan_mesh = pymesh.boolean(electrode_mesh, surface_mesh, 'difference')
	outer_diff = pymesh.boolean(outer_hull, electrode_tan_mesh, 'difference')
	conditioned_surface = pymesh.merge_meshes((outer_diff, electrode_tan_mesh))

	# Generate the surface with the electrode on
	face_id = np.arange(conditioned_surface.num_faces)
	conditioned_surface = pymesh.remove_duplicated_vertices(conditioned_surface)[0] # Remove any duplicate vertices
	surface_with_electrode = pymesh.submesh(conditioned_surface, np.isin(face_id, pymesh.detect_self_intersection(conditioned_surface)[:, 0], invert=True), 0) # Get rid of the duplicate faces on the tangent surface, without merging the points


skin_stl = pymesh.load_mesh('skin_fixed.stl')
skin_stl.enable_connectivity()  # Important in order for orient_electrode to work

p_i_base_vcc = skin_stl.vertices[point]
cr_base_vcc = orient_electrode(skin_stl, p_i_base_vcc)

elec_base_vcc = pymesh.generate_cylinder(
	p_i_base_vcc - (width*cr_base_vcc)/4., p_i_base_vcc + (width*cr_base_vcc)/4., radius, radius, elements)
mrg = pymesh.merge_meshes((skin_stl, elec_base_vcc))

pymesh.save_mesh('skin_elec.stl', mrg)
