import os
import meshio
import pymesh
import numpy as np

import Meshing.mesh_operations as mesh_ops
import Meshing.electrode_operations as electrode_operations

def phm_model_meshing(base_path: str, suffix_name: str, electrode_attributes: dict, max_radius=5):
	##### Import th model files to create the mesh
	print("Importing files") # INFO log
	skin_stl = pymesh.load_mesh(os.path.join(base_path, 'skin' + suffix_name))
	skull_stl = pymesh.load_mesh(os.path.join(base_path, 'skull' + suffix_name))
	csf_stl = pymesh.load_mesh(os.path.join(base_path, 'csf' + suffix_name))
	gm_stl = pymesh.load_mesh(os.path.join(base_path, 'gm' + suffix_name))
	wm_stl = pymesh.load_mesh(os.path.join(base_path, 'wm' + suffix_name))
	ventricles_stl = pymesh.load_mesh(os.path.join(base_path, 'ventricles' + suffix_name))
	cerebellum_stl = pymesh.load_mesh(os.path.join(base_path, 'cerebellum' + suffix_name))
	##### Import th model files to create the mesh

	##### Electrode placement
	print("Placing Electrodes") # INFO log
	skin_stl.enable_connectivity()

	# Add the standard 10-20 electrodes
	elec_array = electrode_operations.standard_electrode_positioning(electrode_attributes['names'], electrode_attributes['coordinates'], skin_stl, electrode_attributes['width'], electrode_attributes['radius'], electrode_attributes['elements'])

	elec_meshes = pymesh.merge_meshes((e_mesh['mesh'] for e_mesh in elec_array.values()))
	sub_outer = electrode_operations.add_electrode(skin_stl, elec_meshes)
	##### Electrode placement

	print("Boundary - Electrode Separation") # INFO log
	sub_outer[0] = pymesh.merge_meshes((sub_outer[0], skull_stl, csf_stl, wm_stl, gm_stl, ventricles_stl, cerebellum_stl))

	if pymesh.detect_self_intersection(sub_outer[0]):
		print("Self-intersections detected") # WARNING log
		sub_outer[0] = pymesh.resolve_self_intersection(sub_outer[0])
	
	part_model = pymesh.tetrahedralize(sub_outer[0], max_radius)

	sp_tet = pymesh.tetrahedralize(sub_outer[1], max_radius)

	boundary_surfaces_elec = pymesh.form_mesh(sp_tet.vertices, sp_tet.faces)
	boundary_surfaces = pymesh.form_mesh(part_model.vertices, part_model.faces)
	boundary_surfaces = pymesh.separate_mesh(boundary_surfaces)

	boundary_surfaces = mesh_ops.boundary_order([skin_stl, skull_stl, csf_stl, gm_stl, wm_stl, ventricles_stl, cerebellum_stl], boundary_surfaces)

	mesh_domains_elec = mesh_ops.domain_extract(part_model, boundary_surfaces_elec, cary_zeros=False, keep_zeros=True)

	dom_roi = [roi['dom_roi'] for roi in elec_array.values()]
	elecs_l, st_srf = electrode_operations.electrodes_separate(part_model, mesh_domains_elec, dom_roi)

	extraction_direction = [['in', 6], ['in', 5], ['out', 1], ['out', 2], ['out', 3], ['out', 4]]
	zero_operations = [[True, False], [False, True], [True, False], [True, False], [False, True], [True, False]]
	output_order = {
		'Cerebellum': 5,
		'Ventricles': 2,
		'Skin': 0,
		'Skull': 1,
		'CSF': 2,
		'GM': 3,
		'WM': 4,
	}

	print("Mesh Forming") # INFO log
	msh_f = mesh_ops.mesh_form([st_srf, boundary_surfaces], extraction_direction, zero_operations, output_order)
	msh_f = mesh_ops.mesh_conditioning(msh_f) # Remove duplicate elements

	# Save the mesh is VTK format
	print("Mesh Saving") # INFO log

	em_c = np.empty(part_model.num_voxels)
	em_p = np.empty(part_model.num_vertices)

	for key in msh_f.keys():
		em_c[msh_f[key]['mesh'].get_attribute("ori_voxel_index").astype(np.int32)] = msh_f[key]['id']
		em_p[np.unique(part_model.voxels[msh_f[key]['mesh'].get_attribute("ori_voxel_index").astype(np.int32)])] = msh_f[key]['id']

	i = 10
	for elec in elecs_l:
		em_c[elec.get_attribute("ori_voxel_index").astype(np.int32)] = i
		em_p[np.unique(part_model.voxels[elec.get_attribute("ori_voxel_index").astype(np.int32)])] = i
		i = i + 1

	mss = meshio.Mesh(
		points=part_model.vertices,
		point_data={"node_groups": em_p},
		cells={"tetra": part_model.voxels},
		cell_data={"mat_id": em_c},
	)

	return {'mesh': mss}
