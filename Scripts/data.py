for i in range(0, sep[0].num_faces):
	adj_voxel = model_tet.get_face_adjacent_voxels(i)
	
	if adj_voxel.shape[0] > 0:
		voxel = model_tet.voxels[i]