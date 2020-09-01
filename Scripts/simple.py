import pymesh
import numpy as np

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
elements = 50


sph_stl = pymesh.load_mesh('spheres_1.stl')
pid = np.array([0, 0, -width])

p_i_base_vcc = electrode_position_sphere(scalp_radius, theta_base_vcc)
cr_base_vcc = orient_electrode(sph_stl, p_i_base_vcc, pid)

p_i_base_gnd = electrode_position_sphere(scalp_radius, theta_base_gnd)
cr_base_gnd = orient_electrode(sph_stl, p_i_base_gnd, pid)

p_i_df_vcc = electrode_position_sphere(scalp_radius, theta_df_vcc)
cr_df_vcc = orient_electrode(sph_stl, p_i_df_vcc, pid)

p_i_df_gnd = electrode_position_sphere(scalp_radius, theta_df_gnd)
cr_df_gnd = orient_electrode(sph_stl, p_i_df_gnd, pid)


#pid = pid/np.linalg.norm(pid)
#cr = orient_electrode(sph_stl, p_i, pid)

elec_base_vcc = pymesh.generate_cylinder(p_i_base_vcc - (width*cr_base_vcc)/4., p_i_base_vcc + (width*cr_base_vcc)/4., radius, radius, elements)

elec_base_gnd = pymesh.generate_cylinder(p_i_base_gnd - (width*cr_base_gnd)/4., p_i_base_gnd + (width*cr_base_gnd)/4., radius, radius, elements)

elec_df_vcc = pymesh.generate_cylinder(p_i_df_vcc - (width*cr_df_vcc)/4., p_i_df_vcc + (width*cr_df_vcc)/4., radius, radius, elements)

elec_df_gnd = pymesh.generate_cylinder(p_i_df_gnd - (width*cr_df_gnd)/4., p_i_df_gnd + (width*cr_df_gnd)/4., radius, radius, elements)

#mrg = pymesh.merge_meshes((sph_stl, elec))
'''
elec_base_vcc = pymesh.boolean(elec_base_vcc, sph_stl, 'union')
elec_base_gnd = pymesh.boolean(elec_base_gnd, sph_stl, 'difference')
elec_df_vcc = pymesh.boolean(elec_df_vcc, sph_stl, 'difference')
elec_df_gnd = pymesh.boolean(elec_df_gnd, sph_stl, 'difference')
'''

s2 = pymesh.load_mesh('spheres_2.stl')
s3 = pymesh.load_mesh('spheres_3.stl')
s4 = pymesh.load_mesh('spheres_4.stl')

mrg = pymesh.merge_meshes((sph_stl, elec_base_vcc, elec_base_gnd, elec_df_vcc, elec_df_gnd, s2, s3, s4))

mrg = pymesh.merge_meshes((sph_stl, elec_base_vcc))
tet = pymesh.tetrahedralize(sph_stl, 10)
tet1 = pymesh.tetrahedralize(elec_base_vcc, 2)

pymesh.save_mesh('sph.stl', elec_base_vcc)
