import os
import Meshing.MeshOperations as MeshOps

import scipy.io as sio

base_path = '/home/dimitris/tetgen1.6.0/build'
# folders = next(os.walk(base_path))[1]
#folders = ['103414', '105014', '105115', '110411', '111716', '113619', '117122', '163129', '196750']
folders = ['105014']

if __name__ == '__main__':
    for folder in folders:
        print("############")
        print("Model " + folder)
        print("############\n")
        standard_electrodes = sio.loadmat(os.path.join(base_path, folder, '10-20_elec_' + folder + '.mat'))
        elec_attributes = {
            'names': [name[0][0] for name in standard_electrodes['ElectrodeNames']],
            'coordinates': standard_electrodes['ElectrodePts'],
            'width': 3,
            'radius': 4,
            'elements': 200,
        }

        meshing = MeshOps.MeshOperations()
        meshing.phm_model_meshing(os.path.join(base_path, folder), '_fixed.stl', 'first_poly_try.poly', elec_attributes)
