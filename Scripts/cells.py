ms = pymesh.form_mesh(model_tet.vertices, model_tet.faces)
sep = pymesh.separate_mesh(ms)

d = pymesh.signed_distance_to_mesh(sep[2], model_tet.vertices)

face_m = pymesh.submesh(ms, model_tet.faces[np.where(d[0] < 0)[0].tolist()], 1)
#vox_m = pymesh.submesh(model_tet, model_tet.faces[np.where(d[0] < 0)[0].tolist()], 1)
vox_m = pymesh.submesh(model_tet, sep[2].faces, 0)
#vox_m = pymesh.submesh(model_tet, sep[2].get_attribute("ori_elem_index").astype(np.int32), 0)

############
"""
* Get the signed distance
* Find the indices of the vertices from to corresponding to model_tet
* Find the faces in d[1] and correlate those with the original index in model_tet
* Use the closest face index as a boundary from the separated meshes
* d = pymesh.signed_distance_to_mesh(model_tet, model_tet.vertices)
* Correlate s_t with f_index and keep only the values that are in both
"""
d = pymesh.signed_distance_to_mesh(model_tet, model_tet.vertices)
f_index = d[1]
domains = []

def domain_extract(mesh, closest_faces, boundary_1, boundary_2):
	s_1 = boundary_1.get_attribute("ori_elem_index")
	s_2 = boundary_2.get_attribute("ori_elem_index")
	s_t = np.unique(np.hstack((s_1, s_2))).astype(np.int32)
	
	selected_faces = np.intersect1d(closest_faces, s_t) # Find common indices
	
	vert_id = np.isin(closest_faces, selected_faces)
	vert_id = np.where(vert_id == True)[0]
	
	vox_id = np.isin(mesh.voxels, vert_id)
	vox_id = np.where(vox_id == True)[0]
	vox_id = np.unique(vox_id)
	
	return pymesh.submesh(mesh, vox_id, 0)

for i in range(0, model_tet.num_components, 2):
	s_1 = sep[i].get_attribute("ori_elem_index")
	s_2 = sep[i + 1].get_attribute("ori_elem_index")
	s_t = np.unique(np.hstack((s_1, s_2))).astype(np.int32)
	
	selected_faces = np.intersect1d(f_index, s_t) # Find common indices
	
	vert_id = np.isin(f_index, selected_faces)
	vert_id = np.where(vert_id == True)[0]
	
	vox_id = np.isin(model_tet.voxels, vert_id)
	vox_id = np.where(vox_id == True)[0]
	vox_id = np.unique(vox_id)
	
	new_m = pymesh.submesh(model_tet, vox_id, 0)
	
	domains.append(new_m)

# Get the vertices from model_tet.vertices where the selected faces has values
############

# Get the face and voxel index
f_i = face_m.get_attribute("ori_face_index")[np.where(face_m.get_attribute("ring") == 0)[0].tolist()].astype(np.int32)
v_i = vox_m.get_attribute("ori_voxel_index")[np.where(vox_m.get_attribute("ring") == 0)[0].tolist()].astype(np.int32)

fc = model_tet.faces[f_i.tolist()]
vx = model_tet.voxels[v_i.tolist()]

u_fc = np.unique(fc)
u_vx = np.unique(vx)

u_vert = np.unique(np.hstack((u_fc, u_vx)))

vt = np.zeros((np.amax(u_vert), 3))
vt = model_tet.vertices[np.amin(u_vert):np.amax(u_vert), :]
#vt = model_tet.vertices[u_vert, :]

#u_vert = model_tet.vertices[u_vert.tolist()]

mod = pymesh.form_mesh(vt, fc, vx)


#m = pymesh.submesh(model_tet, ms.faces)
#vox_m = pymesh.submesh(model_tet, model_tet.faces[np.where(d[0] < 0)[0].tolist()], 1)
#new_m = pymesh.form_mesh(vox_m.vertices, vox_m.faces, vox_m.voxels[np.where(vox_m.get_attribute("ring") == 0)[0].tolist()])