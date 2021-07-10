import os
import yaml
import multiprocessing

import NiiPHM.NiiMesh as NiiMesh

# with open(os.path.realpath('/mnt/c/Users/Dimitris/Desktop/Neuro/Jupyter/sim_settings.yml')) as stream:
with open(os.path.realpath(r'C:\Users\Dimitris\Desktop\Neuro\Jupyter\sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

index_array_name = 'cell_scalars'
base_path = r'D:\Neuro Publication\All_models_meshed\VTK'
output_dir = r'D:\Neuro Publication\All_models_meshed\Nifti'
files = next(os.walk(base_path))[2]
# files = ['fem_1.2-8.vtk']
# structures_to_remove = [1, 2, 3, 7]
structures_to_remove = []
conductivities = [1/a['conductivity'] for a in settings['SfePy']['real_brain']['regions'].values()]

def nfts(fl):
# for fl in files:
    global conductivities
    base_path = r'D:\Neuro Publication\All_models_meshed\VTK'
    index_array_name = 'cell_scalars'
    structures_to_remove = []
    output_dir = r'D:\Neuro Publication\All_models_meshed\Nifti'
    model_id = os.path.splitext(fl)[0].split('_')[-1].split('-')[0].split('.')[0]
    print("Converting model {}".format(model_id))
    nifti = NiiMesh.NiiMesh(os.path.join(base_path, fl), index_array_name, [1, 8])
    _ = nifti.assign_intensities(assign_index=False, unwanted_region_ids=structures_to_remove, assign_values_per_region=True, values_to_assign=conductivities)
    _ = nifti.generate_uniform_grid(15, 0.75, )
    nifti.generate_nifti(image_path=os.path.join(output_dir, 'PHM_' + model_id + '.nii'))


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    pool.map(nfts, files)
