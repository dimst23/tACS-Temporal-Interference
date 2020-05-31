import pymesh
import meshio

import mesh_operations
import gmsh_write

MAX_RADIUS = 5
base_path = '/mnt/c/Users/Dimitris/Desktop/Thes/model/fixed/'

# Import th model files to create the mesh
print("Importing files") # INFO log
skin_stl = pymesh.load_mesh(base_path + 'skin_fixed.stl')
skull_stl = pymesh.load_mesh(base_path + 'skull_fixed.stl')
csf_stl = pymesh.load_mesh(base_path + 'csf_fixed.stl')
gm_stl = pymesh.load_mesh(base_path + 'gm_fixed.stl')
wm_stl = pymesh.load_mesh(base_path + 'wm_fixed.stl')
ventricles_stl = pymesh.load_mesh(base_path + 'ventricles_fixed.stl')
cerebellum_stl = pymesh.load_mesh(base_path + 'cerebellum_fixed.stl')

# Generate the mesh of the model
print("Merging meshes") # INFO log
model = pymesh.merge_meshes((skin_stl, skull_stl, csf_stl, gm_stl, wm_stl, ventricles_stl, cerebellum_stl))

# Generate the tetrahedrals
print("Generating volume") # INFO log
model_tet = pymesh.tetrahedralize(model, MAX_RADIUS)
pymesh.save_mesh('model.mesh', model_tet)

# Separate the boundary surfaces in the generated mesh
print("Mesh boundary surface separation") # INFO log
boundary_surfaces = pymesh.form_mesh(model_tet.vertices, model_tet.faces)
boundary_surfaces = pymesh.separate_mesh(boundary_surfaces)

# Get mesh domains
boundary_order = [4, 6, 1, 2, 3, 5]
extraction_direction = ['in', 'in', 'out', 'out', 'out', 'out']
zero_operations = [[True, False], [False, True], [True, False], [True, False], [False, True], [True, False]]
output_order = [5, 6, 0, 1, 2, 3, 4]

mesh_domains = mesh_operations.mesh_form([model_tet, boundary_surfaces], boundary_order, extraction_direction, zero_operations, output_order)
mesh_domains = mesh_operations.mesh_conditioning(mesh_domains) # Remove duplicate elements

# Write the mesh to gmsh file
physical_tags = [[1, 'Skin'], [2, 'Skull'], [3, 'CSF'], [4, 'GM'], [5, 'WM'], [6, 'Cerebellum'], [7, 'Ventricles']]
bounding_surface_tag = [1, 2, 3, 4, 5, 6, 7]

gmsh_write.gmsh_write(boundary_surfaces, mesh_domains, physical_tags, bounding_surface_tag, 'model.msh')
