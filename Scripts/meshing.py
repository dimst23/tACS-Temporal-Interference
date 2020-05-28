import pymesh

# Import th model files to create the mesh
skin_stl = pymesh.load_mesh('skin_fixed.stl')
skull_stl = pymesh.load_mesh('skull_fixed.stl')
csf_stl = pymesh.load_mesh('csf_fixed.stl')
gm_stl = pymesh.load_mesh('gm_fixed.stl')
wm_stl = pymesh.load_mesh('wm_fixed.stl')
ventricles_stl = pymesh.load_mesh('ventricles_fixed.stl')
cerebellum_stl = pymesh.load_mesh('cerebellum_fixed.stl')

# Generate the boolean objects
skin = pymesh.CSGTree({
	"difference": [
		{"mesh": skin_stl},
		{"mesh": skull_stl}
	]
}).mesh

skull = pymesh.CSGTree({
	"difference": [
		{"mesh": skull_stl},
		{"mesh": csf_stl}
	]
}).mesh

csf = pymesh.CSGTree({
	"union": [
		{"difference": [
			{"difference": [{"mesh": csf_stl}, {"mesh": gm_stl}]},
			{"mesh": cerebellum_stl}
		]},
		{"mesh": ventricles_stl}
	]
}).mesh

brain_grey_matter = pymesh.CSGTree({
	"difference": [
		{"difference": [
			{"mesh": gm_stl},
			{"mesh": wm_stl}
		]},
		{"mesh": ventricles_stl}
	]
}).mesh

brain_white_matter = pymesh.CSGTree({
	"difference": [
		{"mesh": wm_stl},
		{"mesh": ventricles_stl}
	]
}).mesh

csf_new = pymesh.CSGTree({
	"union": [
		{"difference": [
			{"difference": [{"mesh": csf_stl}, {"mesh": gm_stl}]},
			{"mesh": cerebellum_stl}
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

faces = np.zeros((model_mesh.voxels.shape[0], 1))
faces_size = np.zeros((model_mesh.voxels.shape[0], 1))
for i in range(0, model_mesh.voxels.shape[0]):
	adj = model_mesh.get_voxel_adjacent_faces(i)
	if adj.shape[0] != 0:
		faces[i] = adj[0]
		faces_size[i] = adj.shape[0]


# Generate the mesh of the model
model = pymesh.merge_meshes( (skin, skull, csf, brain_grey_matter, brain_white_matter, cerebellum_stl) )

# Fix the model
model_fixed = pymesh.resolve_self_intersection(model)
model_fixed = pymesh.remove_duplicated_faces(model_fixed[0])
model_fixed = pymesh.remove_degenerated_triangles(model_fixed[0])
#model_fixed = pymesh.resolve_self_intersection(model_fixed[0])

# Generate the tetrahedrals
model_tet = pymesh.tetrahedralize(model_fixed, 10)

"""
brain_new = pymesh.CSGTree({
	"difference": [
		{"difference": [
			{"mesh": gm},
			{"mesh": wm}
		]},
		{"mesh": ventricles}
	]
})

csf_new = pymesh.CSGTree({
	"union": [
		{"difference": [
			{"difference": [{"mesh": csf}, {"mesh": gm}]},
			{"mesh": cerebellum}
		]},
		{"mesh": ventricles}
	]
})


model = pymesh.CSGTree({
	"difference": [
		{"mesh": skin},
		{"difference": [
			{"mesh": skull},
			{"difference": [
				{csf_new},
				{brain_new}
			]}
		]}
	]
})
"""

header = '''$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
3
3 1 "Skin"
3 2 "Skull"
3 3 "CSF"
$EndPhysicalNames
$Entities
0 0 3 3
1 2.0 2.0 2.0 180.986481 206.047211 158.320435 0 0
2 20.999016 10.024086 7.278244 162.024994 196.89151 152.314697 0 0
3 23.014862 15.032524 8.441536 160.01503 192.033051 146.313049 0 0
1 2.0 2.0 2.0 180.986481 206.047211 158.320435 1 1 1 1
2 20.999016 10.024086 7.278244 162.024994 196.89151 152.314697 1 2 1 2
3 23.014862 15.032524 8.441536 160.01503 192.033051 146.313049 1 3 1 3
$EndEntities'''

$Nodes
  numEntityBlocks(size_t) numNodes(size_t)
    minNodeTag(size_t) maxNodeTag(size_t)
  entityDim(int) entityTag(int) parametric(int; 0 or 1)
    numNodesInBlock(size_t)
    nodeTag(size_t)
    ...
    x(double) y(double) z(double)
       < u(double; if parametric and entityDim >= 1) >
       < v(double; if parametric and entityDim >= 2) >
       < w(double; if parametric and entityDim == 3) >
    ...
  ...
$EndNodes

$Elements
  numEntityBlocks(size_t) numElements(size_t)
    minElementTag(size_t) maxElementTag(size_t)
  entityDim(int) entityTag(int) elementType(int; see below)
    numElementsInBlock(size_t)
    elementTag(size_t) nodeTag(size_t) ...
    ...
  ...
$EndElements

f_mesh = model.mesh
out_mesh = pymesh.tetrahedralize(f_mesh, 10)

pymesh.save_mesh('gen_file_new.msh', out_mesh, ascii=true)


csf_vents = pymesh.CSGTree({
	"union": [
		{"mesh": csf_stl},
		{"mesh": ventricles_stl}
	]
}).mesh
mrgs = pymesh.merge_meshes((skin_stl, skull_stl, csf_stl, gm_stl, wm_stl, ventricles_stl, cerebellum_stl)) 