import pymesh
import numpy as np

import Library.mesh_operations as mesh_ops
import Library.electrode_operations as electrode_operations
import Library.gmsh_write as gmsh_write

def electrode_position_sphere(radius, theta, phi=0):
	return np.array([radius*np.cos(np.deg2rad(phi))*np.cos(np.deg2rad(theta)), radius*np.cos(np.deg2rad(phi))*np.sin(np.deg2rad(theta)), radius*np.sin(np.deg2rad(phi))])

def orient_electrode(mesh, init_point, delta_point):
	dist = pymesh.signed_distance_to_mesh(mesh, init_point)
	face = dist[1][0]
	#
	# Create a vector with the direction of the face
	p_1 = mesh.vertices[mesh.faces[face][0]]
	p_2 = mesh.vertices[mesh.faces[face][1]]
	dir_vector = p_1 - p_2
	dir_vector = dir_vector/np.linalg.norm(dir_vector)
	
	normal = np.cross(delta_point, dir_vector)
	return normal/np.linalg.norm(normal)

def add_electrode(surface_mesh, electrode_mesh):
	# Get the surface outline including the electrode
	model = pymesh.merge_meshes((surface_mesh, electrode_mesh))
	outer_hull = pymesh.compute_outer_hull(model)
	#
	# Create the surface with the electrode mesh imprinted
	electrode_tan_mesh = pymesh.boolean(electrode_mesh, surface_mesh, 'difference')
	outer_diff = pymesh.boolean(outer_hull, electrode_tan_mesh, 'difference')
	conditioned_surface = pymesh.merge_meshes((outer_diff, electrode_tan_mesh))
	#
	# Generate the surface with the electrode on
	face_id = np.arange(conditioned_surface.num_faces)
	conditioned_surface = pymesh.remove_duplicated_vertices(conditioned_surface)[0]  # Remove any duplicate vertices
	#
	return [pymesh.submesh(conditioned_surface, np.isin(face_id, pymesh.detect_self_intersection(conditioned_surface)[:, 0], invert=True), 0), outer_diff]  # Get rid of the duplicate faces on the tangent surface, without merging the points

MAX_RADIUS = 5
#base_path = './'
base_path = '/mnt/d/Thesis/Tests/model/fixed/'

##### Import th model files to create the mesh
print("Importing files") # INFO log
s1_stl = pymesh.load_mesh(base_path + 'spheres_1.stl')
s2_stl = pymesh.load_mesh(base_path + 'spheres_2.stl')
s3_stl = pymesh.load_mesh(base_path + 'spheres_3.stl')
s4_stl = pymesh.load_mesh(base_path + 'spheres_4.stl')
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
scalp_radius = 85
radius = 4
width = 3
elements = 100
##### Electrode parameters
##### Electrodes

pid = np.array([0, 0, -width])

p_i_base_vcc = electrode_position_sphere(scalp_radius, theta_base_vcc)
cr_base_vcc = orient_electrode(s1_stl, p_i_base_vcc, pid)

p_i_base_gnd = electrode_position_sphere(scalp_radius, theta_base_gnd)
cr_base_gnd = orient_electrode(s1_stl, p_i_base_gnd, pid)

p_i_df_vcc = electrode_position_sphere(scalp_radius, theta_df_vcc)
cr_df_vcc = orient_electrode(s1_stl, p_i_df_vcc, pid)

p_i_df_gnd = electrode_position_sphere(scalp_radius, theta_df_gnd)
cr_df_gnd = orient_electrode(s1_stl, p_i_df_gnd, pid)


elec_base_vcc = pymesh.generate_cylinder(p_i_base_vcc - (width*cr_base_vcc)/4., p_i_base_vcc + (width*cr_base_vcc)/4., radius, radius, elements)

elec_base_gnd = pymesh.generate_cylinder(p_i_base_gnd - (width*cr_base_gnd)/4., p_i_base_gnd + (width*cr_base_gnd)/4., radius, radius, elements)

elec_df_vcc = pymesh.generate_cylinder(p_i_df_vcc - (width*cr_df_vcc)/4., p_i_df_vcc + (width*cr_df_vcc)/4., radius, radius, elements)

elec_df_gnd = pymesh.generate_cylinder(p_i_df_gnd - (width*cr_df_gnd)/4., p_i_df_gnd + (width*cr_df_gnd)/4., radius, radius, elements)

# Generate the meshes and separate the electrodes
elec_meshes = pymesh.merge_meshes((elec_base_vcc, elec_base_gnd, elec_df_vcc, elec_df_gnd))
sub_outer = add_electrode(s1_stl, elec_meshes)

sub_outer[0] = pymesh.merge_meshes((sub_outer[0], s2_stl, s3_stl, s4_stl))
part_model = pymesh.tetrahedralize(sub_outer[0], MAX_RADIUS)

sp_tet = pymesh.tetrahedralize(sub_outer[1], MAX_RADIUS)

boundary_surfaces_elec = pymesh.form_mesh(sp_tet.vertices, sp_tet.faces)
boundary_surfaces = pymesh.form_mesh(part_model.vertices, part_model.faces)
boundary_surfaces = pymesh.separate_mesh(boundary_surfaces)
boundary_surfaces[0] = boundary_surfaces_elec

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

msh_f = mesh_ops.mesh_form([st_srf, boundary_surfaces], [2, 1, 3], ['out', 'out', 'in'], [[True, False], [True, False], [False, True]], [0, 2, 1, 3])

# Save the mesh is VTK format
import meshio

em_c = np.empty(part_model.num_voxels)
em_t = np.empty(part_model.num_voxels)
em_p = np.empty(part_model.num_vertices)

i = 1
for mesh in msh_f:
	em_c[mesh.get_attribute("ori_voxel_index").astype(np.int32)] = i
	em_t[mesh.get_attribute("ori_voxel_index").astype(np.int32)] = 20
	em_p[np.unique(part_model.voxels[mesh.get_attribute("ori_voxel_index").astype(np.int32)])] = i
	i = i + 1

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
