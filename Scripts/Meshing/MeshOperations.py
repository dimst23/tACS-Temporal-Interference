import os
import meshio
import pymesh
import numpy as np

import Meshing.ElectrodeOperations as ElecOps
import FileOps.FileOperations as FileOps

class MeshOperations(ElecOps.ElectrodeOperations):
    def __init__(self):
        self.merged_meshes = None
        print()

    def phm_model_meshing(self, base_path: str, suffix_name: str, mesh_filename: str, electrode_attributes: dict):
        ##### Import th model files to create the mesh
        print("Importing files") # INFO log
        skin_stl = pymesh.load_mesh(os.path.join(base_path, 'skin' + suffix_name))
        skull_stl = pymesh.load_mesh(os.path.join(base_path, 'skull' + suffix_name))
        csf_stl = pymesh.load_mesh(os.path.join(base_path, 'csf' + suffix_name))
        gm_stl = pymesh.load_mesh(os.path.join(base_path, 'gm' + suffix_name))
        wm_stl = pymesh.load_mesh(os.path.join(base_path, 'wm' + suffix_name))
        ventricles_stl = pymesh.load_mesh(os.path.join(base_path, 'ventricles' + suffix_name))
        cerebellum_stl = pymesh.load_mesh(os.path.join(base_path, 'cerebellum' + suffix_name))
        ##### Import th model files to create the mesh

        ##### Electrode placement
        print("Placing Electrodes") # INFO log
        electrodes_object = ElecOps.ElectrodeOperations(skin_stl, electrode_attributes)
        electrodes_object.standard_electrode_positioning(width=electrode_attributes['width'], radius=electrode_attributes['radius'], elements=electrode_attributes['elements'])
        skin_with_electrodes = electrodes_object.add_electrodes_on_skin()[0]

        electrodes_single_mesh = electrodes_object.get_electrode_single_mesh()
        electrodes = pymesh.boolean(electrodes_single_mesh, skin_stl, 'difference')
        pymesh.map_face_attribute(electrodes_single_mesh, electrodes, 'face_sources')
        ##### Electrode placement

        self.merged_meshes = pymesh.merge_meshes((skin_with_electrodes, skull_stl, csf_stl, gm_stl, wm_stl, cerebellum_stl, ventricles_stl))
        regions = self.region_points([skin_stl, skull_stl, csf_stl, gm_stl, wm_stl, cerebellum_stl, ventricles_stl], electrodes, 0.1)

        FileOps.FileOperations.poly_write(mesh_filename, self.merged_meshes.vertices, self.merged_meshes.faces, self.merged_meshes.get_attribute('face_sources'), regions)

    def region_points(self, boundary_surfaces: list, electrode_mesh, shift: float, max_volume=30):
        points = {}
        i = 1
        for surface in boundary_surfaces:
            vertices = surface.vertices
            max_z_point_id = np.where(vertices[:, 2] == np.amax(vertices[:, 2]))[0][0]
            points[str(i)] = {
                'coordinates': np.round(vertices[max_z_point_id] - np.absolute(np.multiply(vertices[max_z_point_id]/np.linalg.norm(vertices[max_z_point_id]), [0, 0, shift])), 6),
                'max_volume': max_volume,
            }
            i = i + 1

        i = 10
        electrode_regions = electrode_mesh.get_attribute('face_sources')
        electrode_mesh.add_attribute('vertex_valance')
        vertex_valance = electrode_mesh.get_attribute('vertex_valance')
        for region in np.unique(electrode_regions):
            faces = electrode_mesh.faces[np.where(electrode_regions == region)[0]]
            vertices = electrode_mesh.vertices[np.unique(faces)]
            max_valance_point = np.where(vertex_valance[np.unique(faces)] == np.amax(vertex_valance[np.unique(faces)]))[0][0]
            points[str(i)] = {
                'coordinates': np.round(vertices[max_valance_point] - np.multiply(vertices[max_valance_point]/np.linalg.norm(vertices[max_valance_point]), shift), 6),
                'max_volume': max_volume,
            }
            i = i + 1

        return points
