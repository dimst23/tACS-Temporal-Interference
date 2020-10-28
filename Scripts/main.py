import os
import Meshing.phm_model_meshing as model_meshing

import meshio
import scipy.io as sio

base_path = '/mnt/c/Users/Dimitris/Documents/Brains/'
folders = next(os.walk(base_path))[1]

if __name__ == '__main__':
	for folder in folders:
		standard_electrodes = sio.loadmat(os.path.join(base_path, folder, '10-20_elec_' + folder + '.mat'))
		elec_attributes = {
			'names': [name[0][0] for name in standard_electrodes['ElectrodeNames']],
			'coordinates': standard_electrodes['ElectrodePts'],
			'width': 3,
			'radius': 4,
			'elements': 200,
		}

		mesh = model_meshing.phm_model_meshing(os.path.join(base_path, folder), '_fixed.stl', elec_attributes)
		mesh['mesh'].write(os.path.join(base_path, folder, 'mesh_' + folder + '_10-20.vtk'))
