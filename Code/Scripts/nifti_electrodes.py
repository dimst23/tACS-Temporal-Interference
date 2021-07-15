import os
import gc
import yaml
import numpy as np
import pyvista as pv
import pandas as pd
import nibabel as nib
import PVGeo

base_path ='/mnt/d/Neuro Publication/All_models_meshed/VTK'
# save_path = '/mnt/c/Users/Dimitris/Desktop/Neuro/e_field/electrodes'
save_path = '/mnt/e/HPC_Models/e_field/electrodes'
npz_path = '/mnt/e/HPC_Models/NPZ'
files = np.sort(next(os.walk(npz_path))[2])

with open(os.path.realpath('/mnt/c/Users/Dimitris/Documents/repos/ttis-software/Code/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

distance_margin = 15
voxel_size = 1

ommit_models = ['101309', '103111', '103414', '105014', '105115', '110411', '111716', '113619', '117122', '118932', '120111', '122317']

for fl in files:
    model_id = os.path.splitext(fl)[0].split('_')[0]
    if model_id in ommit_models:
        continue
    print(model_id)
    msh = pv.UnstructuredGrid(os.path.join(base_path, 'Meshed_AAL_10-10_' + model_id + '.vtk'))

    mesh_pts = msh.threshold(value=[1, 8], scalars='cell_scalars', preference='cell').cell_centers()
    pts_loc = np.isin(mesh_pts['cell_scalars'], [4, 5, 6])
    mesh_points = mesh_pts.points[pts_loc]

    bounds = msh.bounds
    x_bounds_init = np.ceil(np.sum(np.abs(bounds[0:2]) + distance_margin)/voxel_size).astype(np.int32)
    y_bounds_init = np.ceil(np.sum(np.abs(bounds[2:4]) + distance_margin)/voxel_size).astype(np.int32)
    z_bounds_init = np.ceil(np.sum(np.abs(bounds[4:6]) + distance_margin)/voxel_size).astype(np.int32)

    grid = pv.UniformGrid((x_bounds_init, y_bounds_init, z_bounds_init))
    grid.origin = -1*np.array([np.sum(np.abs(bounds[0:2]))/2., np.sum(np.abs(bounds[2:4]))/2., np.sum(np.abs(bounds[4:6]))/2.]) - distance_margin
    grid.spacing = [voxel_size]*3

    img_header = nib.Nifti1Header()
    img_header.set_xyzt_units('mm', 'sec')

    npz_arrays = np.load(os.path.join(npz_path, fl), allow_pickle=True)
    fields = npz_arrays['e_field']
    electrodes = list(settings['SfePy']['electrodes']['10-10-mod'].items())

    for electrode in electrodes:
        electrode_id = electrode[1]['id'] - 10
        electrode_name = electrode[0]

        data = {'x': mesh_points[:, 0], 'y': mesh_points[:, 1], 'z': mesh_points[:, 2], 'int_x': fields[electrode_id, :, 0], 'int_y': fields[electrode_id, :, 1], 'int_z': fields[electrode_id, :, 2]}
        df = pd.DataFrame(data, columns=['x', 'y', 'z', 'int_x', 'int_y', 'int_z'])
        grid_points = PVGeo.points_to_poly_data(df)

        voxels = grid.interpolate(grid_points, radius=voxel_size*4, sharpness=8)

        for e_field_array in voxels.array_names:
            voxel_data = voxels[e_field_array].reshape((z_bounds_init, y_bounds_init, x_bounds_init)).transpose()
            direction = e_field_array.split('_')[1]

            x_minus_bound = bounds[0] - distance_margin
            y_minus_bound = bounds[2] - distance_margin
            z_minus_bound = bounds[4] - distance_margin
            affine = np.array([[voxel_size, 0, 0, x_minus_bound], [0, voxel_size, 0, y_minus_bound], [0, 0, voxel_size, z_minus_bound], [0, 0, 0, 1]])

            try:
                os.makedirs(os.path.join(save_path, model_id, electrode_name))
            except OSError:
                pass

            img = nib.Nifti1Image(voxel_data, affine, img_header)
            nib.save(img, os.path.join(save_path, model_id, electrode_name, 'Efield_{}_{}_{}.nii'.format(direction, electrode_name, model_id)))

        del voxels
        del voxel_data
        gc.collect()

    del msh
    del npz_arrays
    del fields
    gc.collect()
