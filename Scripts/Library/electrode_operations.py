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
	point_id = np.where(np.sum(mesh.vertices == init_point, axis=1))[0][0] # Unique point assumed
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
	"""[summary]

	Args:
		surface_mesh ([type]): [description]
		electrode_mesh ([type]): [description]

	Returns:
		[type]: [description]
	"""
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
	#
	return [pymesh.submesh(conditioned_surface, np.isin(face_id, pymesh.detect_self_intersection(conditioned_surface)[:, 0], invert=True), 0), outer_diff]  # Get rid of the duplicate faces on the tangent surface, without merging the points

def electrode_separate(mesh, bounding_roi):
	"""Separate the electrodes from the surface of the head

	Args:
		mesh (pymesh.Mesh.Mesh): Complete mesh of the object to separate the electrodes
		bounding_roi (dict): Dictionary containing the bounds in X, Y, Z plane for the box of the electrode

	Returns:
		list: The first element contains the submesh with the ROI data and the second contains the rest mesh elements
	"""
	# Bounding ROI
	vert_x = np.logical_and(mesh.vertices[:, 0] >= bounding_roi['x_min'], mesh.vertices[:, 0] <= bounding_roi['x_max'])
	vert_y = np.logical_and(mesh.vertices[:, 1] >= bounding_roi['y_min'], mesh.vertices[:, 1] <= bounding_roi['y_max'])
	vert_z = np.logical_and(mesh.vertices[:, 2] >= bounding_roi['z_min'], mesh.vertices[:, 2] <= bounding_roi['z_max'])
	
	# Get the ROI vertex indices
	vert_id_roi = np.arange(mesh.num_vertices)
	roi_ids = (vert_x * vert_y * vert_z > 0)

	# Calculate the resulting voxels ROI
	vox_id_roi = np.isin(mesh.voxels, vert_id_roi[roi_ids])
	vox_id_roi = np.where(vox_id_roi == True)[0]
	vox_id_roi = np.unique(vox_id_roi)

	# Calculate the rest voxels
	vox_id_rest = np.isin(mesh.voxels, vert_id_roi[roi_ids], invert=True)
	vox_id_rest = np.where(vox_id_rest == True)[0]
	vox_id_rest = np.unique(vox_id_rest)

	return [pymesh.submesh(mesh, vox_id_rest, 0), pymesh.submesh(mesh, vox_id_rest, 0)]
