import os
import pymesh
import Meshing.MeshOperations as MeshOps

import scipy.io as sio

base_path = '/home/dimitris/tetgen1.6.0/models'
base_path2 = '/home/dimitris/Documents/Thesis/10-10/'
# folders = next(os.walk(base_path))[1]
folders = ['103414', '105014', '105115', '110411', '111716', '113619', '117122', '163129', '196750']
#folders = ['105115', '110411', '111716', '113619', '117122', '163129', '196750']
#folders = ['126325']

if __name__ == '__main__':
    for folder in folders:
        print("############")
        print("Model " + folder)
        print("############\n")
        standard_electrodes = sio.loadmat(os.path.join(base_path, folder, '10-10_elec_' + folder + '.mat'))
        elec_attributes = {
            'names': [name[0][0] for name in standard_electrodes['ElectrodeNames']],
            'coordinates': standard_electrodes['ElectrodePts'],
            'width': 4,
            'radius': 4,
            'elements': 200,
        }

        skin_stl = pymesh.load_mesh(os.path.join(base_path, folder, 'skin_fixed.stl'))

        meshing = MeshOps.MeshOperations(skin_stl, elec_attributes)
        meshing.load_surface_meshes(os.path.join(base_path, folder), ['skin_fixed.stl', 'skull_fixed.stl', 'csf_fixed.stl', 'gm_fixed.stl', 'wm_fixed.stl', 'cerebellum_fixed.stl', 'ventricles_fixed.stl'])
        meshing.phm_model_meshing(os.path.join(base_path2, 'meshed_model_10-10_' + folder + '.poly'))
