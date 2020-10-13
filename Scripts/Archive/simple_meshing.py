import pymesh
import meshio

import Library.mesh_operations as mesh_operations
import Library.gmsh_write as gmsh_write
import numpy as np

MAX_RADIUS = 5
base_path = '/mnt/d/Thesis/Tests/model/fixed/'

# Import th model files to create the mesh
print("Importing files") # INFO log
s1_stl = pymesh.load_mesh(base_path + 'spheres_1.stl')
s2_stl = pymesh.load_mesh(base_path + 'spheres_2.stl')
s3_stl = pymesh.load_mesh(base_path + 'spheres_3.stl')
s4_stl = pymesh.load_mesh(base_path + 'spheres_4.stl')

# Generate the mesh of the model
print("Merging meshes") # INFO log
#model = pymesh.merge_meshes((s1_stl, s2_stl, s3_stl))

##### Electrodes
s1_stl.enable_connectivity()
def electrode_position_sphere(radius, theta, phi=0):
	return np.array([radius*np.cos(np.deg2rad(phi))*np.cos(np.deg2rad(theta)), radius*np.cos(np.deg2rad(phi))*np.sin(np.deg2rad(theta)), radius*np.sin(np.deg2rad(phi))])

def orient_electrode(mesh, init_point, delta_point):
	dist = pymesh.signed_distance_to_mesh(mesh, init_point)
	face = dist[1][0]
	
	# Create a vector with the direction of the face
	p_1 = mesh.vertices[mesh.faces[face][0]]
	p_2 = mesh.vertices[mesh.faces[face][1]]
	dir_vector = p_1 - p_2
	dir_vector = dir_vector/np.linalg.norm(dir_vector)
	
	normal = np.cross(delta_point, dir_vector)
	return normal/np.linalg.norm(normal)

#import scipy.spatial as scp
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

scalp_radius = 85
radius = 4
width = 3
elements = 100

pid = np.array([0, 0, -width])

p_i_base_vcc = electrode_position_sphere(scalp_radius, theta_base_vcc)
cr_base_vcc = orient_electrode(s1_stl, p_i_base_vcc, pid)

p_i_base_gnd = electrode_position_sphere(scalp_radius, theta_base_gnd)
cr_base_gnd = orient_electrode(s1_stl, p_i_base_gnd, pid)
'''
p_i_df_vcc = electrode_position_sphere(scalp_radius, theta_df_vcc)
cr_df_vcc = orient_electrode(s1_stl, p_i_df_vcc, pid)

p_i_df_gnd = electrode_position_sphere(scalp_radius, theta_df_gnd)
cr_df_gnd = orient_electrode(s1_stl, p_i_df_gnd, pid)
'''

elec_base_vcc = pymesh.generate_cylinder(p_i_base_vcc - (width*cr_base_vcc)/4., p_i_base_vcc + (width*cr_base_vcc)/4., radius, radius, elements)

elec_base_gnd = pymesh.generate_cylinder(p_i_base_gnd - (width*cr_base_gnd)/4., p_i_base_gnd + (width*cr_base_gnd)/4., radius, radius, elements)
'''
elec_df_vcc = pymesh.generate_cylinder(p_i_df_vcc - (width*cr_df_vcc)/4., p_i_df_vcc + (width*cr_df_vcc)/4., radius, radius, elements)

elec_df_gnd = pymesh.generate_cylinder(p_i_df_gnd - (width*cr_df_gnd)/4., p_i_df_gnd + (width*cr_df_gnd)/4., radius, radius, elements)
'''

#electrode_resolution = 0.4

elec_base_vcc = pymesh.boolean(elec_base_vcc, s1_stl, 'difference')
#spl_mesh, info = pymesh.split_long_edges(elec_base_vcc, 0.2)
#elec_base_vcc, info = pymesh.collapse_short_edges(spl_mesh, 0.2, preserve_feature=True)

#elec_base_vcc = pymesh.tetrahedralize(elec_base_vcc, electrode_resolution)

#elec_base_gnd = pymesh.generate_cylinder(p_i_base_gnd - (width*cr_base_gnd)/4., p_i_base_gnd + (width*cr_base_gnd)/4., radius, radius, elements)

#elec_base_gnd = pymesh.boolean(elec_base_gnd, s1_stl, 'difference')
#spl_mesh, info = pymesh.split_long_edges(elec_base_gnd, 0.2)
#elec_base_gnd, info = pymesh.collapse_short_edges(spl_mesh, 0.2, preserve_feature=True)

pymesh.save_mesh('ts.stl', elec_base_vcc)
pymesh.save_mesh('ts_2.stl', s1_stl)

#elec_base_gnd = pymesh.tetrahedralize(elec_base_gnd, electrode_resolution)

#boundary_surfaces.append(pymesh.form_mesh(elec_base_vcc.vertices, elec_base_vcc.faces))
#boundary_surfaces.append(pymesh.form_mesh(elec_base_gnd.vertices, elec_base_gnd.faces))

#mesh_domains.append(elec_base_vcc)
#mesh_domains.append(elec_base_gnd)
#### Electrodes
#model = pymesh.merge_meshes((elec_base_gnd, elec_base_vcc, s1_stl, s2_stl, s3_stl))
model = pymesh.merge_meshes((elec_base_gnd, s1_stl, s2_stl, s3_stl))
#model = pymesh.merge_meshes((s1_stl, s2_stl, s3_stl))


# Generate the tetrahedrals
print("Generating volume") # INFO log
model_tet = pymesh.tetrahedralize(model, MAX_RADIUS)
pymesh.save_mesh('model.mesh', model_tet)

#mdl_fnl = pymesh.merge_meshes((elec_base_gnd, elec_base_vcc, s1_stl, s2_stl, s3_stl))

# Separate the boundary surfaces in the generated mesh
print("Mesh boundary surface separation") # INFO log
boundary_surfaces = pymesh.form_mesh(model_tet.vertices, model_tet.faces)
boundary_surfaces = pymesh.separate_mesh(boundary_surfaces)

# Get mesh domains
boundary_order = [2, 1, 0]
extraction_direction = ['in', 'out', 'out']
zero_operations = [[True, False], [False, True], [True, False]]
output_order = [2, 0, 1]

mesh_domains = mesh_operations.mesh_form([model_tet, boundary_surfaces], boundary_order, extraction_direction, zero_operations, output_order)
mesh_domains = mesh_operations.mesh_conditioning(mesh_domains) # Remove duplicate elements

# Write the mesh to gmsh file
physical_tags = [[1, 'Skin'], [2, 'Skull'], [3, 'CSF'], [4, 'VCC'], [5, 'GND']]
bounding_surface_tag = [1, 2, 3, 4, 5]


gmsh_write.gmsh_write(boundary_surfaces, mesh_domains, physical_tags, bounding_surface_tag, 'model.msh')

#### SfePy file
mrg = pymesh.merge_meshes((model_tet, elec_base_vcc, elec_base_gnd))

em_c = np.empty(mrg.num_voxels)
em_p = np.empty(mrg.num_vertices)

i = 1
for domain in mesh_domains:
	try:
		em_c[domain.get_attribute("ori_voxel_index").astype(np.int32)] = i
		em_p[np.unique(mrg.voxels[domain.get_attribute("ori_voxel_index").astype(np.int32)])] = i
		i += 1
	except:
		pass

em_c[np.where(mrg.get_attribute("voxel_sources") == 1)[0]] = i
em_p[np.where(mrg.get_attribute("vertex_sources") == 1)[0]] = i
i += 1
em_c[np.where(mrg.get_attribute("voxel_sources") == 2)[0]] = i
em_p[np.where(mrg.get_attribute("vertex_sources") == 2)[0]] = i

'''
mss = meshio.Mesh(
    points=mrg.vertices,
    cells=[{"tetra": mrg.voxels}, {"triangle": mrg.faces}],
	cell_data={"gmsh:physical": em, "gmsh:physical2": em},
	cell_sets={"Skin": mesh_domains[0].get_attribute("ori_voxel_index").astype(np.int32), 
               "Skull": mesh_domains[1].get_attribute("ori_voxel_index").astype(np.int32), 
               "CSF": mesh_domains[2].get_attribute("ori_voxel_index").astype(np.int32), 
               "GM": np.where(mrg.get_attribute("voxel_sources") == 1)[0], 
               "WM": np.where(mrg.get_attribute("voxel_sources") == 2)[0],
              },
)
'''
mss = meshio.Mesh(
    points=mrg.vertices,
	point_data={"node_groups": em_p},
    cells={"tetra": mrg.voxels},
	cell_data={"mat_id": em_c},
)

mss.write('msh.vtk')
#### SfePy File
