import os
from numpy.core.defchararray import index
import yaml
import pyvista as pv
import numpy as np

import NiiPHM.NiiMesh as NiiMesh

with open(os.path.realpath('/mnt/c/Users/Dimitris/Desktop/Neuro/Jupyter/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

index_array_name = 'mat_id'
base_path = '/mnt/d/Neuro Publication/Mesh_Convergence/101309/8mm'
output_dir = '/mnt/d/Neuro Publication/Mesh_Convergence/101309/8mm/nii'
files = next(os.walk(base_path))[2]
files = ['fem_1.2-8.vtk']
# structures_to_remove = [1, 2, 3, 7]
structures_to_remove = []
conductivities = [a['conductivity'] for a in settings['SfePy']['real_brain']['regions'].values()]

for fl in files:
    model_id = os.path.splitext(fl)[0].split('_')[-1].split('-')[0]
    print("Converting model {}".format(model_id))
    nifti = NiiMesh.NiiMesh(os.path.join(base_path, fl), index_array_name, [1, 8])
    _ = nifti.assign_intensities(assign_index=False, unwanted_region_ids=structures_to_remove, assign_values_per_region=True, values_to_assign=conductivities)
    _ = nifti.generate_uniform_grid(15, 0.75, )
    nifti.generate_nifti()
