import pymesh
import numpy as np

import Library.mesh_operations as mesh_ops
import Library.electrode_operations as electrode_operations

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
skin_stl = pymesh.load_mesh(base_path + 'skin_fixed.stl')
##### Import th model files to create the mesh

# Generate the mesh of the model
print("Merging meshes") # INFO log
#model = pymesh.merge_meshes((s1_stl, s2_stl, s3_stl))

##### Electrode placement
width = 3
radius = 4
elements = 100

skin_stl.enable_connectivity()

p_i_base_vcc = skin_stl.vertices[10421]
cr_base_vcc = electrode_operations.orient_electrode(skin_stl, p_i_base_vcc)

p_i_base_gnd = skin_stl.vertices[3187]
cr_base_gnd = electrode_operations.orient_electrode(skin_stl, p_i_base_gnd)

elec_base_vcc = pymesh.generate_cylinder(p_i_base_vcc - (width*cr_base_vcc)/4., p_i_base_vcc + (width*cr_base_vcc)/4., radius, radius, elements)

elec_base_gnd = pymesh.generate_cylinder(p_i_base_gnd - (width*cr_base_gnd)/4., p_i_base_gnd + (width*cr_base_gnd)/4., radius, radius, elements)

elec_meshes = pymesh.merge_meshes((elec_base_vcc, elec_base_gnd))
#sub_outer = add_electrode(skin_stl, elec_base_vcc)
sub_outer = add_electrode(skin_stl, elec_meshes)

# pymesh.save_mesh('br_elec.stl', sub_outer[0])
##### Electrode placement

part_model = pymesh.tetrahedralize(sub_outer[0], MAX_RADIUS)
sp_tet = pymesh.tetrahedralize(sub_outer[1], MAX_RADIUS)

boundary_surfaces = pymesh.form_mesh(sp_tet.vertices, sp_tet.faces)
#boundary_surfaces = pymesh.separate_mesh(boundary_surfaces)
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

mesh_domains_1 = mesh_ops.domain_extract(part_model, boundary_surfaces)

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
