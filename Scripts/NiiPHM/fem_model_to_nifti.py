import os
import sys
import yaml
import pyvista as pv
import PVGeo
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn.image

with open(os.path.realpath(r'C:\Users\Dimitris\Desktop\Neuro\Jupyter\sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

index_array_name = 'cell_scalars'
base_path = r'D:\Neuro Publication\Preliminary_Models\Unsolved'
# files = next(os.walk(base_path))[2]
files = ['meshed_model_10-20_101309.1.vtk']

for fl in files:
	model_id = os.path.splitext(fl)[0].split('_')[-1].split('.')[0]
	print("Converting model {}".format(model_id))
	mesh = pv.UnstructuredGrid(os.path.join(base_path, fl))
	mesh_pts = mesh.threshold(value=[1, 8], scalars=index_array_name, preference='cell').cell_centers()

	setting_values = list(settings['SfePy']['real_brain']['regions'].values())
	conductivities = np.zeros(mesh_pts[index_array_name].shape)

	# structures_to_remove = [1, 2, 3, 7]
	structures_to_remove = []

	for id in np.unique(mesh_pts[index_array_name]):
		if id in structures_to_remove:
			conductivities[np.where(mesh_pts[index_array_name] == id)[0]] = 0
		else:
			conductivities[np.where(mesh_pts[index_array_name] == id)[0]] = 1./setting_values[id - 1]['conductivity']
			# conductivities[np.where(mesh_pts[index_array_name] == id)[0]] = 1

	data = {'x': mesh_pts.points[:, 0], 'y': mesh_pts.points[:, 1], 'z': mesh_pts.points[:, 2], 'a': conductivities}
	# data = {'x': mesh_pts.points[:, 0], 'y': mesh_pts.points[:, 1], 'z': mesh_pts.points[:, 2],'a': mesh_pts[index_array_name]}
	df = pd.DataFrame(data, columns=['x', 'y', 'z', 'a'])
	grid_points = PVGeo.points_to_poly_data(df)
	
	margin = 15
	voxel_size = 0.75 # Voxel size
	bounds = mesh.bounds
	x_bounds_init = np.ceil(np.sum(np.abs(bounds[0:2]) + margin)/voxel_size).astype(np.int32)
	y_bounds_init = np.ceil(np.sum(np.abs(bounds[2:4]) + margin)/voxel_size).astype(np.int32)
	z_bounds_init = np.ceil(np.sum(np.abs(bounds[4:6]) + margin)/voxel_size).astype(np.int32)
	
	grid = pv.UniformGrid((x_bounds_init, y_bounds_init, z_bounds_init))
	grid.origin = -1*np.array([np.sum(np.abs(bounds[0:2]))/2., np.sum(np.abs(bounds[2:4]))/2., np.sum(np.abs(bounds[4:6]))/2.]) - margin
	grid.spacing = [voxel_size]*3
	
	radius_multiplier = 4
	vox = grid.interpolate(grid_points, radius=voxel_size*radius_multiplier, sharpness=8)
	
	
	x_minus_bound = bounds[0] - margin
	y_minus_bound = bounds[2] - margin
	z_minus_bound = bounds[4] - margin
	affine = np.array([[voxel_size, 0, 0, x_minus_bound], [0, voxel_size, 0, y_minus_bound], [0, 0, voxel_size, z_minus_bound], [0, 0, 0, 1]])
	
	img_header = nib.Nifti1Header()
	img_header.set_xyzt_units('mm', 'sec')
	voxel_data = vox['a'].reshape((z_bounds_init, y_bounds_init, x_bounds_init)).transpose()
	
	img = nib.Nifti1Image(voxel_data, affine, img_header)
	nib.save(img, os.path.join(r'D:\Neuro Publication\Preliminary_Models\Nifti_Models', 'PHM_' + model_id + '.nii'))
