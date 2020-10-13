import pymesh
import numpy as np

def domain_extract(mesh, boundary, direction='out', cary_zeros=True, keep_zeros=False):
	"""Extract the subdomains of a mesh defined by a low bound surface

	Arguments:
		mesh {[type]} -- [description]
		boundary {[type]} -- [description]

	Keyword Arguments:
		direction {str} -- [description] (default: {'out'})
		cary_zeros {bool} -- [description] (default: {True})
		keep_zeros {bool} -- [description] (default: {False})

	Returns:
		[type] -- [At index 0 the current domain can be found. At index 1 the complementary mesh is saved.]
	"""
	distances = list(pymesh.signed_distance_to_mesh(boundary, mesh.vertices))

	# The following line has been added after a lot of pain and effort, lasting for more than 3 days! At first it seemed that for some reason the function was keeping the points that we supposed to belong on the other surface. After some accidental printing of the distances to manually check it, I noticed a negative distance but it was -1e-15!!! A distance such small it is zero, but due to round issues it was considered as a negative one!! The solution is to round to a float and maybe a smaller rounding is more than sufficient.
	distances[0] = np.round(distances[0], 8)
	
	if direction == 'out':
		vert_id_roi = np.where(distances[0] > 0)[0]
		vert_id_rest = np.where(distances[0] < 0)[0]
	elif direction == 'in':
		vert_id_roi = np.where(distances[0] < 0)[0]
		vert_id_rest = np.where(distances[0] > 0)[0]
	else:
		print("Wrong '%s' direction entered" % direction)
		return -1
	
	# Get the region of interest voxels
	vox_id_roi = np.isin(mesh.voxels, vert_id_roi)
	vox_id_roi = np.where(vox_id_roi == True)[0]
	vox_id_roi = np.unique(vox_id_roi)
	
	# Save the rest of the voxels
	vox_id_rest = np.isin(mesh.voxels, vert_id_rest)
	vox_id_rest = np.where(vox_id_rest == True)[0]
	vox_id_rest = np.unique(vox_id_rest)
	
	if cary_zeros:
		vox_id_rest_0 = np.isin(mesh.voxels, np.where(distances[0] == 0)[0])
		vox_id_rest_0 = np.where(np.sum(vox_id_rest_0, axis=1) == 4)[0]
		vox_id_rest = np.unique(np.hstack((vox_id_rest, vox_id_rest_0)))
	
	if keep_zeros:
		vox_id_roi_0 = np.isin(mesh.voxels, np.where(distances[0] == 0)[0])
		vox_id_roi_0 = np.where(np.sum(vox_id_roi_0, axis=1) == 4)[0]
		vox_id_roi = np.unique(np.hstack((vox_id_roi, vox_id_roi_0)))
	
	if vox_id_rest.size == 0:
		return [pymesh.submesh(mesh, vox_id_roi, 0), 0]
	else:
		return [pymesh.submesh(mesh, vox_id_roi, 0), pymesh.submesh(mesh, vox_id_rest, 0)]

def mesh_form(meshes: list, extract_directions: list, zero_operations: list, output_order: dict):
	"""[summary]

	Arguments:
		meshes {list} -- The list of bounding surfaces. At zero index the initial surface resides
		boundary_order {list} -- Order that the boundary surfaces shall be used in sequence
		extract_directions {list} -- Directions of extraction in accordance with the domain_extract function
		zero_operations {list} -- What to do with the zero voxels
		output_order {list} -- Correct order the elements shall be in the output

	Returns:
		list -- A list of ordered, based on output_order, conditioned meshes
	"""
	domains = {}
	last_id = 0
	for i in range(0, len(meshes[1]) - 1):
		if i == 0:
			# For the first mesh start from the base
			dm = domain_extract(meshes[0], meshes[1][i + 1], direction=extract_directions[i], cary_zeros=zero_operations[i][0], keep_zeros=zero_operations[i][1])
		else:
			dm = domain_extract(dm[1], meshes[1][i + 1], direction=extract_directions[i], cary_zeros=zero_operations[i][0], keep_zeros=zero_operations[i][1])
		#domains.insert(output_order[i], dm[0])
		domains[output_order[i]] = {'mesh': dm[0], 'id': i}
		last_id = i + 1

	if type(dm[1]) is not int:
		#domains.insert(output_order[-1], dm[1])
		domains[output_order[-1]] = {'mesh': dm[1], 'id': last_id}
	return domains

def mesh_conditioning(meshes: list):
	"""[summary]

	Arguments:
		meshes {list} -- [description]

	Returns:
		list -- [description]
	"""
	for i in range(0, len(meshes)):
		for j in range(i + 1, len(meshes)):
			dups = np.where(np.isin(meshes[j].get_attribute("ori_voxel_index"), meshes[i].get_attribute("ori_voxel_index"), invert=True))[0]
			if dups.shape[0] > 0:
				meshes[j] = pymesh.submesh(meshes[j], dups, 0)
	return meshes

def boundary_order(init_boundary: list, boundary_surfaces: list):
	ordered_bounds = []
	for boundary in init_boundary:
		for surface in boundary_surfaces:
			distances = pymesh.distance_to_mesh(surface, boundary.vertices)
			# Arbitrary number of non zero values
			if distances[0].shape[0] != np.count_nonzero(distances[0]):
				ordered_bounds.append(surface)
	return ordered_bounds
