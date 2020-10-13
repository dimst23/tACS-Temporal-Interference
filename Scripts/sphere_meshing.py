import pymesh
import numpy as np

import Meshing.mesh_operations as mesh_ops
import Meshing.electrode_operations as electrode_operations
import Meshing.gmsh_write as gmsh_write

MAX_RADIUS = 5
#base_path = './'
#base_path = '/mnt/d/Thesis/Tests/model/fixed/'
base_path = '/mnt/c/Users/Dimitris/Nextcloud/Documents/Neuroscience Bachelor Thesis/Public Repository/tacs-temporal-interference/CAD/Sphere/'

##### Import th model files to create the mesh
print("Importing files") # INFO log
#outer_stl = pymesh.load_mesh(base_path + 'spheres_outer.stl')
s1_stl = pymesh.load_mesh(base_path + 'spheres_skin.stl')
s2_stl = pymesh.load_mesh(base_path + 'spheres_skull.stl')
s3_stl = pymesh.load_mesh(base_path + 'spheres_csf.stl')
s4_stl = pymesh.load_mesh(base_path + 'spheres_brain.stl')
##### Import th model files to create the mesh

# Generate the mesh of the model
##### Electrodes
s1_stl.enable_connectivity()

##### Electrode position
# Theta
theta_base_vcc = 258.5217
theta_base_gnd = 326.2893
theta_df_vcc = 101.4783
theta_df_gnd = 33.7107

# Phi
phi_base_vcc = 0
phi_base_gnd = 0
phi_df_vcc = 0
phi_df_gnd = 0
##### Electrode position

##### Electrode parameters
scalp_radius = np.amax(s1_stl.vertices[:, 0])
radius = 4
width = 3
elements = 200
##### Electrode parameters
##### Electrodes

pid = np.array([0, 0, -width])

p_i_base_vcc = electrode_operations.electrode_position_sphere(scalp_radius, theta_base_vcc)
cr_base_vcc = electrode_operations.orient_electrode_sphere(s1_stl, p_i_base_vcc, pid)

p_i_base_gnd = electrode_operations.electrode_position_sphere(scalp_radius, theta_base_gnd)
cr_base_gnd = electrode_operations.orient_electrode_sphere(s1_stl, p_i_base_gnd, pid)

p_i_df_vcc = electrode_operations.electrode_position_sphere(scalp_radius, theta_df_vcc)
cr_df_vcc = electrode_operations.orient_electrode_sphere(s1_stl, p_i_df_vcc, pid)

p_i_df_gnd = electrode_operations.electrode_position_sphere(scalp_radius, theta_df_gnd)
cr_df_gnd = electrode_operations.orient_electrode_sphere(s1_stl, p_i_df_gnd, pid)


elec_base_vcc = pymesh.generate_cylinder(p_i_base_vcc - (width*cr_base_vcc)/4., p_i_base_vcc + (width*cr_base_vcc)/4., radius, radius, elements)

elec_base_gnd = pymesh.generate_cylinder(p_i_base_gnd - (width*cr_base_gnd)/4., p_i_base_gnd + (width*cr_base_gnd)/4., radius, radius, elements)

elec_df_vcc = pymesh.generate_cylinder(p_i_df_vcc - (width*cr_df_vcc)/4., p_i_df_vcc + (width*cr_df_vcc)/4., radius, radius, elements)

elec_df_gnd = pymesh.generate_cylinder(p_i_df_gnd - (width*cr_df_gnd)/4., p_i_df_gnd + (width*cr_df_gnd)/4., radius, radius, elements)

# Generate the meshes and separate the electrodes
elec_meshes = pymesh.merge_meshes((elec_base_vcc, elec_base_gnd, elec_df_vcc, elec_df_gnd))
sub_outer = electrode_operations.add_electrode(s1_stl, elec_meshes)

sub_outer[0] = pymesh.merge_meshes((sub_outer[0], s2_stl, s3_stl, s4_stl))
part_model = pymesh.tetrahedralize(sub_outer[0], MAX_RADIUS)

sp_tet = pymesh.tetrahedralize(sub_outer[1], MAX_RADIUS)

boundary_surfaces_elec = pymesh.form_mesh(sp_tet.vertices, sp_tet.faces)
boundary_surfaces = pymesh.form_mesh(part_model.vertices, part_model.faces)
boundary_surfaces = pymesh.separate_mesh(boundary_surfaces)

boundary_surfaces = mesh_ops.boundary_order([s1_stl, s2_stl, s3_stl, s4_stl], boundary_surfaces)

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
mesh_domains_elec = mesh_ops.domain_extract(part_model, boundary_surfaces_elec)

elecs_l, st_srf = electrode_operations.electrodes_separate(part_model, mesh_domains_elec, [dom_roi_1, dom_roi_2, dom_roi_3, dom_roi_4])

'''
pymesh.save_mesh('st_srf1.stl', boundary_surfaces[0])
pymesh.save_mesh('st_srf2.stl', boundary_surfaces[1])
pymesh.save_mesh('st_srf3.stl', boundary_surfaces[2])
pymesh.save_mesh('st_srf4.stl', boundary_surfaces[3])
'''

#msh_f = mesh_ops.mesh_form([st_srf, boundary_surfaces], [2, 1, 3], ['out', 'out', 'out'], [[True, False], [True, False], [True, False]], [0, 2, 1, 3])
msh_f = mesh_ops.mesh_form([st_srf, boundary_surfaces], ['out', 'out', 'out'], [[True, False], [True, False], [True, False]], ['Skin', 'Skull', 'CSF', 'Brain'])

# Save the mesh is VTK format
import meshio

em_c = np.empty(part_model.num_voxels)
em_p = np.empty(part_model.num_vertices)

i = 4
'''
for mesh in msh_f:
	em_c[mesh.get_attribute("ori_voxel_index").astype(np.int32)] = i
	em_t[mesh.get_attribute("ori_voxel_index").astype(np.int32)] = 20
	em_p[np.unique(part_model.voxels[mesh.get_attribute("ori_voxel_index").astype(np.int32)])] = i
	i = i + 1
'''
for key in msh_f.keys():
	em_c[msh_f[key]['mesh'].get_attribute("ori_voxel_index").astype(np.int32)] = msh_f[key]['id']
	em_p[np.unique(part_model.voxels[msh_f[key]['mesh'].get_attribute("ori_voxel_index").astype(np.int32)])] = msh_f[key]['id']

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

mss.write('msh.vtk')

''' MSH file save
bnd_srf = [pymesh.form_mesh(elecs_l[0].vertices, elecs_l[0].faces), pymesh.form_mesh(elecs_l[1].vertices, elecs_l[1].faces), pymesh.form_mesh(elecs_l[2].vertices, elecs_l[2].faces), pymesh.form_mesh(elecs_l[3].vertices, elecs_l[3].faces), boundary_surfaces[0], boundary_surfaces[2], boundary_surfaces[1]]

msh_dom = [elecs_l[0], elecs_l[1], elecs_l[2], elecs_l[3], msh_f[0], msh_f[1], msh_f[2]]

physical_tags = [[1, 'Skin'], [2, 'Skull'], [3, 'CSF'], [4, 'Base_VCC'], [5, 'Base_GBD'], [6, 'DF_VCC'], [7, 'DF_GND']]
bounding_surface_tag = [1, 2, 3, 4, 5, 6, 7]

gmsh_write.gmsh_write(bnd_srf, msh_dom, physical_tags, bounding_surface_tag, 'model.msh')
'''
