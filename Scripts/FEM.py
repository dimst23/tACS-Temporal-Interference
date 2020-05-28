import pymesh
import numpy as np

def domain_extract(mesh, boundary, direction='out'):
	distances = pymesh.signed_distance_to_mesh(boundary, mesh.vertices)
	
	if direction == 'out':
		vert_id = np.where(distances[0] >= 0)[0]
	elif direction == 'in':
		vert_id = np.where(distances[0] < 0)[0]
	else:
		print("Wrong '%s' direction entered".direction)
		return -1
	
	vox_id = np.isin(mesh.voxels, vert_id)
	vox_id = np.where(vox_id == True)[0]
	vox_id = np.unique(vox_id)
	
	return pymesh.submesh(mesh, vox_id, 0)


# Import th model files to create the mesh
print("Importing files") # INFO log
skin_stl = pymesh.load_mesh('skin_fixed.stl')
skull_stl = pymesh.load_mesh('skull_fixed.stl')
csf_stl = pymesh.load_mesh('csf_fixed.stl')
gm_stl = pymesh.load_mesh('gm_fixed.stl')
wm_stl = pymesh.load_mesh('wm_fixed.stl')
ventricles_stl = pymesh.load_mesh('ventricles_fixed.stl')
cerebellum_stl = pymesh.load_mesh('cerebellum_fixed.stl')

# Generate the boolean objects

# Generate the mesh of the model
print("Merging meshes") # INFO log
#model = pymesh.merge_meshes((skin_stl, skull_stl, csf_stl, gm_stl, wm_stl, ventricles_stl, cerebellum_stl))
model = pymesh.merge_meshes((skin_stl, skull_stl, csf_stl))

# Generate the tetrahedrals
print("Generating volume") # INFO log
model_tet = pymesh.tetrahedralize(model, 5)
#pymesh.save_mesh('gen_file_new.msh', model_tet, ascii=true)

#distance = pymesh.signed_distance_to_mesh(model_tet, model_tet.vertices)
#f_index = distance[1]

ms = pymesh.form_mesh(model_tet.vertices, model_tet.faces)
sep = pymesh.separate_mesh(ms)

new_m = domain_extract(model_tet, sep[0], sep[2])

tet = pymesh.tetgen()
tet.points = model.vertices
tet.triangles = model.faces
tet.verbosity = 1
#tet.point_markers = points
#tet.triangle_markers = faces
tet.run()

model_tet = tet.mesh

