import pymesh
import numpy as np

def domain_extract(mesh, boundary, direction='out'):
	distances = pymesh.signed_distance_to_mesh(boundary, mesh.vertices)
	
	if direction == 'out':
		vert_id_roi = np.where(distances[0] >= 0)[0]
		vert_id_rest = np.where(distances[0] < 0)[0]
	elif direction == 'in':
		vert_id_roi = np.where(distances[0] < 0)[0]
		vert_id_rest = np.where(distances[0] > 0)[0]
	else:
		print("Wrong '%s' direction entered".direction)
		return -1
	
	vox_id_roi = np.isin(mesh.voxels, vert_id_roi)
	vox_id_roi = np.where(vox_id_roi == True)[0]
	vox_id_roi = np.unique(vox_id_roi)
	
	vox_id_rest = np.isin(mesh.voxels, vert_id_rest)
	vox_id_rest = np.where(vox_id_rest == True)[0]
	vox_id_rest = np.unique(vox_id_rest)
	
	return [pymesh.submesh(mesh, vox_id_roi, 0), pymesh.submesh(mesh, vox_id_rest, 0)]

def domain_extract_2(mesh, boundary, direction='out'):
	distances = pymesh.signed_distance_to_mesh(boundary, mesh.vertices)
	
	if direction == 'out':
		vert_id_roi = np.where(distances[0] >= 0)[0]
		vert_id_rest = np.where(distances[0] < 0)[0]
	elif direction == 'in':
		vert_id_roi = np.where(distances[0] < 0)[0]
		vert_id_rest = np.where(distances[0] > 0)[0]
	else:
		print("Wrong '%s' direction entered".direction)
		return -1
	
	vox_id_roi = np.isin(mesh.voxels, vert_id_roi)
	vox_id_roi = np.where(vox_id_roi == True)[0]
	vox_id_roi = np.unique(vox_id_roi)
	
	vox_id_rest = np.isin(mesh.voxels, vert_id_rest)
	vox_id_rest = np.where(vox_id_rest == True)[0]
	vox_id_rest = np.unique(vox_id_rest)
	
	return [pymesh.submesh(mesh, vox_id_roi, 0), pymesh.submesh(mesh, vox_id_rest, 0)]

domains = []
model = model_tet

for i in range(1, len(sep) - 1):
	dm = domain_extract(model, sep[i], direction='out')
	domains.append(dm[0])
	model = dm[1]

# Skin
dm = domain_extract(model_tet, sep[1], direction='out')
skin_domain = dm[0]

# Skull
dm = domain_extract(dm[1], sep[2], direction='out')
skull_domain  = dm[0]

# CSF
temp = pymesh.merge_meshes((sep[3], sep[4])) # Merge Grey Matter and Cerebellum
dom = domain_extract_2(dm[1], temp, direction='out')
csf_domain  = dom[0]

# Grey Matter
temp = pymesh.merge_meshes((sep[5], sep[4])) # Merge White Matter and Ventricles
dm = domain_extract(dm[1], temp, direction='out')
greymatter_domain = dm[0]

#dm_2 = domain_extract(model_tet, sep[3], direction='in')
#temp = pymesh.merge_meshes((sep[5], sep[6])) # Merge White Matter and Ventricles
#dm_2 = domain_extract(dm_2[0], sep[5], direction='out')
#greymatter_domain = dm_2[0]

# White Matter
#tms = pymesh.form_mesh(dm_2[0].vertices, dm_2[0].faces)
#tms = pymesh.form_mesh(greymatter_domain.vertices, greymatter_domain.faces)
#dm_2 = domain_extract(model_tet, tms, direction='in')
#temp = pymesh.merge_meshes((sep[5], sep[6])) # Merge White Matter and Ventricles
dm = domain_extract(dm[1], sep[6], direction='out')
white_matter_domain = dm[0]

# Cerebellum
cerebellum_domain = domain_extract(model_tet, sep[4], direction='in')[0]

# Ventricles
ventricles_domain = domain_extract(model_tet, sep[6], direction='in')[0]

pymesh.save_mesh('skin_domain.msh', skin_domain)
pymesh.save_mesh('skull_domain.msh', skull_domain)
pymesh.save_mesh('greymatter_domain.msh', greymatter_domain)
pymesh.save_mesh('white_matter_domain.msh', white_matter_domain)
pymesh.save_mesh('cerebellum_domain.msh', cerebellum_domain)
pymesh.save_mesh('ventricles_domain.msh', ventricles_domain)

meshio.write_points_cells(
	"skin_domain.vtk",
	skin_domain.vertices,
	[("tetra", skin_domain.voxels)],
)

meshio.write_points_cells(
	"skull_domain.vtk",
	skull_domain.vertices,
	[("tetra", skull_domain.voxels)],
)

meshio.write_points_cells(
	"cerebellum_domain.vtk",
	cerebellum_domain.vertices,
	[("tetra", cerebellum_domain.voxels)],
)

meshio.write_points_cells(
	"ventricles_domain.vtk",
	ventricles_domain.vertices,
	[("tetra", ventricles_domain.voxels)],
)

meshio.write_points_cells(
	"csf_domain.vtk",
	csf_domain.vertices,
	[("tetra", csf_domain.voxels)],
)

skin_domain.num_voxels + skull_domain.num_voxels + csf_domain.num_voxels + greymatter_domain.num_voxels + white_matter_domain.num_voxels + cerebellum_domain.num_voxels + ventricles_domain.num_voxels == model_tet.num_voxels

ids = np.hstack((skin_domain.get_attribute("ori_voxel_index"), skull_domain.get_attribute("ori_voxel_index"), csf_domain.get_attribute("ori_voxel_index"), greymatter_domain.get_attribute("ori_voxel_index"), white_matter_domain.get_attribute("ori_voxel_index"), cerebellum_domain.get_attribute("ori_voxel_index"), ventricles_domain.get_attribute("ori_voxel_index")))

n = pymesh.submesh(model_tet, ids.astype(np.int32), 1)
m = pymesh.submesh(model_tet, n.get_attribute("ori_voxel_index")[np.where(n.get_attribute("ring") == 1)[0]].astype(np.int32), 0)

meshio.write_points_cells(
	"missing.vtk",
	m.vertices,
	[("tetra", m.voxels)],
)

meshio.write_points_cells(
	"dm_1_1.vtk",
	dm[1].vertices,
	[("tetra", dm[1].voxels)],
)

for i in range(0, len(sep)):
	pymesh.save_mesh('sep_' + str(i) + '.stl', sep[i])

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
model_tet = pymesh.tetrahedralize(model, 10)
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

