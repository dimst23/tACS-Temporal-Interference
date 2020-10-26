import pymesh
import numpy as np
import scipy.io as sio

import Meshing.mesh_operations as mesh_ops
import Meshing.electrode_operations as electrode_operations

MAX_RADIUS = 5
#base_path = './'
base_path = '/mnt/d/Thesis/Tests/model/fixed/'

##### Import th model files to create the mesh
print("Importing files") # INFO log
skin_stl = pymesh.load_mesh(base_path + 'skin_fixed.stl')
skull_stl = pymesh.load_mesh(base_path + 'skull_fixed.stl')
csf_stl = pymesh.load_mesh(base_path + 'csf_fixed.stl')
gm_stl = pymesh.load_mesh(base_path + 'gm_fixed.stl')
wm_stl = pymesh.load_mesh(base_path + 'wm_fixed.stl')
ventricles_stl = pymesh.load_mesh(base_path + 'ventricles_fixed.stl')
cerebellum_stl = pymesh.load_mesh(base_path + 'cerebellum_fixed.stl')
##### Import th model files to create the mesh

##### Electrode placement
print("Placing Electrodes") # INFO log
width = 3
radius = 4
elements = 200

skin_stl.enable_connectivity()

'''
p_i_base_vcc = skin_stl.vertices[1078]
cr_base_vcc = electrode_operations.orient_electrode(skin_stl, p_i_base_vcc)

p_i_base_gnd = skin_stl.vertices[2400]
cr_base_gnd = electrode_operations.orient_electrode(skin_stl, p_i_base_gnd)

p_i_df_vcc = skin_stl.vertices[14246]
cr_df_vcc = electrode_operations.orient_electrode(skin_stl, p_i_df_vcc)

p_i_df_gnd = skin_stl.vertices[13539]
cr_df_gnd = electrode_operations.orient_electrode(skin_stl, p_i_df_gnd)

elec_base_vcc = pymesh.generate_cylinder(p_i_base_vcc - (width*cr_base_vcc)/4., p_i_base_vcc + (width*cr_base_vcc)/4., radius, radius, elements)
elec_base_gnd = pymesh.generate_cylinder(p_i_base_gnd - (width*cr_base_gnd)/4., p_i_base_gnd + (width*cr_base_gnd)/4., radius, radius, elements)
elec_df_vcc = pymesh.generate_cylinder(p_i_df_vcc - (width*cr_df_vcc)/4., p_i_df_vcc + (width*cr_df_vcc)/4., radius, radius, elements)
elec_df_gnd = pymesh.generate_cylinder(p_i_df_gnd - (width*cr_df_gnd)/4., p_i_df_gnd + (width*cr_df_gnd)/4., radius, radius, elements)
'''

# Add the standard 10-20 electrodes
standard_electrodes = sio.loadmat('10-20_elec.mat')
e_names = [name[0][0] for name in standard_electrodes['EEGLab3']]
elec_array = electrode_operations.standard_electrode_positioning(e_names, standard_electrodes['EEGPts3'], skin_stl, width, radius, elements)

elec_meshes = pymesh.merge_meshes((e_mesh['mesh'] for e_mesh in elec_array.values()))
sub_outer = electrode_operations.add_electrode(skin_stl, elec_meshes)
##### Electrode placement

print("Boundary - Electrode Separation") # INFO log
sub_outer[0] = pymesh.merge_meshes((sub_outer[0], skull_stl, csf_stl, wm_stl, gm_stl, ventricles_stl, cerebellum_stl))
part_model = pymesh.tetrahedralize(sub_outer[0], MAX_RADIUS)

sp_tet = pymesh.tetrahedralize(sub_outer[1], MAX_RADIUS)

boundary_surfaces_elec = pymesh.form_mesh(sp_tet.vertices, sp_tet.faces)
boundary_surfaces = pymesh.form_mesh(part_model.vertices, part_model.faces)
boundary_surfaces = pymesh.separate_mesh(boundary_surfaces)

boundary_surfaces = mesh_ops.boundary_order([skin_stl, skull_stl, csf_stl, gm_stl, wm_stl, ventricles_stl, cerebellum_stl], boundary_surfaces)

'''
dom_roi_1 = {
	'x_min': np.amin(elec_base_vcc.vertices[:, 0]),
	'x_max': np.amax(elec_base_vcc.vertices[:, 0]),
	'y_min': np.amin(elec_base_vcc.vertices[:, 1]),
	'y_max': np.amax(elec_base_vcc.vertices[:, 1]),
	'z_min': np.amin(elec_base_vcc.vertices[:, 2]),
	'z_max': np.amax(elec_base_vcc.vertices[:, 2]),
}

dom_roi_2 = {
	'x_min': np.amin(elec_base_gnd.vertices[:, 0]),
	'x_max': np.amax(elec_base_gnd.vertices[:, 0]),
	'y_min': np.amin(elec_base_gnd.vertices[:, 1]),
	'y_max': np.amax(elec_base_gnd.vertices[:, 1]),
	'z_min': np.amin(elec_base_gnd.vertices[:, 2]),
	'z_max': np.amax(elec_base_gnd.vertices[:, 2]),
}

dom_roi_3 = {
	'x_min': np.amin(elec_df_vcc.vertices[:, 0]),
	'x_max': np.amax(elec_df_vcc.vertices[:, 0]),
	'y_min': np.amin(elec_df_vcc.vertices[:, 1]),
	'y_max': np.amax(elec_df_vcc.vertices[:, 1]),
	'z_min': np.amin(elec_df_vcc.vertices[:, 2]),
	'z_max': np.amax(elec_df_vcc.vertices[:, 2]),
}

dom_roi_4 = {
	'x_min': np.amin(elec_df_gnd.vertices[:, 0]),
	'x_max': np.amax(elec_df_gnd.vertices[:, 0]),
	'y_min': np.amin(elec_df_gnd.vertices[:, 1]),
	'y_max': np.amax(elec_df_gnd.vertices[:, 1]),
	'z_min': np.amin(elec_df_gnd.vertices[:, 2]),
	'z_max': np.amax(elec_df_gnd.vertices[:, 2]),
}
'''
mesh_domains_elec = mesh_ops.domain_extract(part_model, boundary_surfaces_elec, cary_zeros=False, keep_zeros=True)

dom_roi = [roi['dom_roi'] for roi in elec_array.values()]
elecs_l, st_srf = electrode_operations.electrodes_separate(part_model, mesh_domains_elec, dom_roi)

boundary_order = [4, 6, 1, 2, 3, 5]
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
import meshio
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

mss.write('msh_real_10-20_zeros.vtk')

'''
#elec_sep_vcc = electrode_operations.electrode_separate(mesh_domains_1[1], dom_roi_1)
#elec_sep_gnd = electrode_operations.electrode_separate(elec_sep_vcc[1], dom_roi_2)

# Add any missing points in the original model
#mesh_domains_1[0] = pymesh.submesh(part_model, np.hstack((mesh_domains_1[0].get_attribute('ori_voxel_index').astype(np.int32), elec_sep_gnd[1].get_attribute('ori_voxel_index').astype(np.int32))), 0)

elecs_l, st_srf = electrode_operations.electrodes_separate(part_model, mesh_domains_1, [dom_roi_1, dom_roi_2])

elec_sep_vcc = elecs_l[0]
elec_sep_gnd = elecs_l[1]
mesh_domains_1[0] = st_srf

import meshio

em_c = np.empty(part_model.num_voxels)
em_p = np.empty(part_model.num_vertices)

em_c[mesh_domains_1[0].get_attribute("ori_voxel_index").astype(np.int32)] = 1
em_p[np.unique(part_model.voxels[mesh_domains_1[0].get_attribute("ori_voxel_index").astype(np.int32)])] = 1

em_c[elec_sep_vcc.get_attribute("ori_voxel_index").astype(np.int32)] = 2
em_p[np.unique(part_model.voxels[elec_sep_vcc.get_attribute("ori_voxel_index").astype(np.int32)])] = 2

em_c[elec_sep_gnd.get_attribute("ori_voxel_index").astype(np.int32)] = 3
em_p[np.unique(part_model.voxels[elec_sep_gnd.get_attribute("ori_voxel_index").astype(np.int32)])] = 3

mss = meshio.Mesh(
    points=part_model.vertices,
	point_data={"node_groups": em_p},
    cells={"tetra": part_model.voxels},
	cell_data={"mat_id": em_c},
)

mss.write('msh.vtk')
#pymesh.save_mesh('sphere_elec.msh', part_model)
'''
