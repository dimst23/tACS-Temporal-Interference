skin = pymesh.CSGTree({
	"difference": [
		{"mesh": skin_stl},
		{"mesh": skull_stl}
	]
})

skull = pymesh.CSGTree({
	"difference": [
		skin,
		{"mesh": csf_stl}
	]
})

csf = pymesh.CSGTree({
	"union": [
		{"difference": [
			skull,
			{"union": [
				{"mesh": gm_stl}, 
				{"mesh": cerebellum_stl}
			]},
			
		]},
		{"mesh": ventricles_stl}
	]
})

brain_grey_matter = pymesh.CSGTree({
	"difference": [
		{"difference": [
			csf,
			{"mesh": wm_stl}
		]},
		{"mesh": ventricles_stl}
	]
})

model_new = pymesh.CSGTree({
	"difference": [
		{"mesh": skin_stl},
		{"difference": [
			{"mesh": skull_stl},
			csf_new
		]}
	]
})

outer = pymesh.compute_outer_hull(model_tet, all_layers=True)
parttion = pymesh.partition_into_cells(model_tet)

vert = np.zeros(sep[0].num_vertices)
for i in range(0, sep[0].num_vertices):
	vert[i] = np.where(sep[0].vertices[i] == model_tet.vertices)[0][0]
	
	

import pymesh
import numpy as np

skin_stl = pymesh.load_mesh('skin_fixed.stl')
skull_stl = pymesh.load_mesh('skull_fixed.stl')
csf_stl = pymesh.load_mesh('csf_fixed.stl')

model = pymesh.merge_meshes((skin_stl, skull_stl, csf_stl))

points = np.hstack((np.ones(skin_stl.num_vertices, dtype=np.int32), np.zeros(skull_stl.num_vertices, dtype=np.int32) - 1, np.zeros(csf_stl.num_vertices, dtype=np.int32) - 2))

faces = np.hstack((np.zeros(skin_stl.num_faces, dtype=np.int32), np.zeros(skull_stl.num_faces, dtype=np.int32) - 10, np.zeros(csf_stl.num_faces, dtype=np.int32) - 20))

tet = pymesh.tetgen()
tet.points = model.vertices
tet.triangles = model.faces
tet.verbosity = 1
#tet.point_markers = points
tet.triangle_markers = faces
tet.run()

model_tet = tet.mesh
