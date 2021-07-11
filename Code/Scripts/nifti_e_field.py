import os
import gc
import numpy as np
import pyvista as pv
import pandas as pd
import nibabel as nib
import PVGeo

base_path ='/media/dimitris/Shared/Neuro Publication/All_models_meshed/VTK'
save_path = '/media/dimitris/Shared/Neuro Publication/All_models_meshed/Nifti/e_field'
npz_path = '/media/dimitris/Elements/HPC_Models'
files = np.sort(next(os.walk(npz_path))[2])

distance_margin = 15
voxel_size = 0.75

for fl in files:
    model_id = os.path.splitext(fl)[0].split('_')[0]
    print(model_id)
    msh = pv.UnstructuredGrid(os.path.join(base_path, 'Meshed_AAL_10-10_' + model_id + '.vtk'))

    mesh_pts = msh.threshold(value=[1, 8], scalars='cell_scalars', preference='cell').cell_centers()
    pts_loc = np.isin(mesh_pts['cell_scalars'], [4, 5, 6])

    npz_arrays = np.load(os.path.join(npz_path, fl), allow_pickle=True)
    field = npz_arrays['e_field']
    field_base = field[53] - field[8]
    field_df = field[52] - field[24]

    mesh_points = mesh_pts.points[pts_loc]
    data = {'x': mesh_points[:, 0], 'y': mesh_points[:, 1], 'z': mesh_points[:, 2], 'int_x_b': field_base[:, 0], 'int_y_b': field_base[:, 1], 'int_z_b': field_base[:, 2], 'int_x_d': field_df[:, 0], 'int_y_d': field_df[:, 1], 'int_z_d': field_df[:, 2]}
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'int_x_b', 'int_y_b', 'int_z_b', 'int_x_d', 'int_y_d', 'int_z_d'])
    grid_points = PVGeo.points_to_poly_data(df)

    bounds = msh.bounds
    x_bounds_init = np.ceil(np.sum(np.abs(bounds[0:2]) + distance_margin)/voxel_size).astype(np.int32)
    y_bounds_init = np.ceil(np.sum(np.abs(bounds[2:4]) + distance_margin)/voxel_size).astype(np.int32)
    z_bounds_init = np.ceil(np.sum(np.abs(bounds[4:6]) + distance_margin)/voxel_size).astype(np.int32)

    grid = pv.UniformGrid((x_bounds_init, y_bounds_init, z_bounds_init))
    grid.origin = -1*np.array([np.sum(np.abs(bounds[0:2]))/2., np.sum(np.abs(bounds[2:4]))/2., np.sum(np.abs(bounds[4:6]))/2.]) - distance_margin
    grid.spacing = [voxel_size]*3

    img_header = nib.Nifti1Header()
    img_header.set_xyzt_units('mm', 'sec')

    voxels = grid.interpolate(grid_points, radius=voxel_size*4, sharpness=8)

    for e_field_array in voxels.array_names:
        voxel_data = voxels[e_field_array].reshape((z_bounds_init, y_bounds_init, x_bounds_init)).transpose()
        direction = e_field_array.split('_')[1]
        e_field_montage = 'base' if e_field_array.split('_')[2] == 'b' else 'df'

        x_minus_bound = bounds[0] - distance_margin
        y_minus_bound = bounds[2] - distance_margin
        z_minus_bound = bounds[4] - distance_margin
        affine = np.array([[voxel_size, 0, 0, x_minus_bound], [0, voxel_size, 0, y_minus_bound], [0, 0, voxel_size, z_minus_bound], [0, 0, 0, 1]])

        try:
            os.mkdir(os.path.join(save_path, model_id))
        except OSError:
            pass

        img = nib.Nifti1Image(voxel_data, affine, img_header)
        nib.save(img, os.path.join(save_path, model_id, 'Efield_{}_{}_{}.nii'.format(direction, e_field_montage, model_id)))

    del msh
    del npz_arrays
    del field
    del voxel_data
    del voxels
    gc.collect()
