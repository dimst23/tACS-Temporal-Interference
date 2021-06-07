import os
import yaml
import pymesh
import Meshing.MeshOperations as MeshOps

import scipy.io as sio

# base_path = '/home/dimitris/tetgen1.6.0/models'
base_path = '/home/dimitris/Documents/Thesis/STL_Models/'
base_path2 = '/home/dimitris/Documents/Thesis/Models with Electrodes/meshed/'
base_path3 = '/home/dimitris/Documents/Thesis/Models with Electrodes/'
# folders = next(os.walk(base_path3))[1]
folders = ['106016', '110411', '118932']

with open(os.path.realpath('/mnt/d/Neuro Publication/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

if __name__ == '__main__':
    for folder in folders:
        if folder == 'meshed':
            continue
        print("############")
        print("Model " + folder)
        print("############\n")
        standard_electrodes = sio.loadmat(os.path.join(base_path3, folder, '10-20_elec_' + folder + '.mat'))
        elec_attributes = {
            'names': [name[0][0] for name in standard_electrodes['ElectrodeNames']],
            'coordinates': standard_electrodes['ElectrodePts'],
            'ids': settings['SfePy']['electrodes']['10-20'],
            'width': 4,
            'radius': 4,
            'elements': 200,
        }

        skin_stl = pymesh.load_mesh(os.path.join(base_path, folder, 'skin_fixed.stl'))

        meshing = MeshOps.MeshOperations(skin_stl, elec_attributes)
        meshing.load_surface_meshes(os.path.join(base_path, folder), ['skin_fixed.stl', 'skull_fixed.stl', 'csf_fixed.stl', 'gm_fixed.stl', 'wm_fixed.stl', 'cerebellum_fixed.stl', 'ventricles_fixed.stl'])
        meshing.phm_model_meshing(os.path.join(base_path2, 'meshed_model_10-20_' + folder + '.poly'))
