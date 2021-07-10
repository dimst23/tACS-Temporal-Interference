import os
import yaml
import pymesh
import Meshing.MeshOperations as MeshOps

import scipy.io as sio
import numpy as np

# base_path = '/home/dimitris/tetgen1.6.0/models'
base_path = '/home/dimitris/Documents/Thesis/STL_Models/'
base_path2 = '/home/dimitris/Documents/Thesis/Models_with_Electrodes/'
# base_path3 = '/home/dimitris/Documents/Thesis/Models_with_Electrodes/'
base_path3 = '/media/dimitris/Shared/Thesis/Electrodes_for_models/'
folders = np.sort(next(os.walk(base_path))[1])
# folders = ['106016', '118932', '120111', '122317', '122620', '125525']
# folders = ['101309']
electrodes_to_omit=['Nz', 'N2', 'AF10', 'F10', 'FT10', 'T10(M2)', 'TP10', 'PO10', 'I2', 'Iz', 'I1', 'PO9', 'TP9', 'T9(M1)', 'FT9', 'F9', 'AF9', 'N1', 'P10']

with open(os.path.realpath('/home/dimitris/repos/tacs-temporal-interference/Scripts/FEM/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

if __name__ == '__main__':
    for folder in folders:
        if folder == 'meshed':
            continue
        print("############")
        print("Model " + folder)
        print("############\n")
        standard_electrodes = sio.loadmat(os.path.join(base_path3, '10-10_elec_' + folder + '.mat'))
        elec_attributes = {
            'names': [name[0][0] for name in standard_electrodes['ElectrodeNames']],
            'coordinates': standard_electrodes['ElectrodePts'],
            'ids': settings['SfePy']['electrodes']['10-10-mod'],
            'width': 4,
            'radius': 4,
            'elements': 200,
        }

        skin_stl = pymesh.load_mesh(os.path.join(base_path, folder, 'skin_fixed.stl'))

        meshing = MeshOps.MeshOperations(skin_stl, elec_attributes)
        meshing.load_surface_meshes(os.path.join(base_path, folder), ['skin_fixed.stl', 'skull_fixed.stl', 'csf_fixed.stl', 'gm_fixed.stl', 'wm_fixed.stl', 'cerebellum_fixed.stl', 'ventricles_fixed.stl'])
        meshing.phm_model_meshing(os.path.join(base_path2, 'meshed_model_10-10_' + folder + '.poly'), electrodes_to_omit=electrodes_to_omit)
