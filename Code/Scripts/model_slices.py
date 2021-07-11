import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

mesh = pv.UnstructuredGrid(r'D:\Neuro Publication\All_models_meshed\VTK\Meshed_AAL_10-10_101309.vtk')

nifti = nib.load(r'D:\Neuro Publication\All_models_meshed\Nifti\PHM_101309.nii')
affine = nifti.affine
mesh_nifti = pv.UniformGrid((nifti.shape[0], nifti.shape[1], nifti.shape[2]))
mesh_nifti.origin = np.array([affine[0, 3], affine[1, 3], affine[2, 3]])
mesh_nifti.spacing = [affine[0, 0]]*3
mesh_nifti.point_arrays['structures'] = nifti.get_fdata().transpose().flatten()

cmap = plt.cm.get_cmap("viridis", 7)

#slices = mesh.threshold(value=[1, 8], scalars='cell_scalars', preference='cell').slice_orthogonal()
#slices.plot(cmap=cmap)

slices_nifti = mesh_nifti.slice_orthogonal(x=-7, y=-8, z=-6)
slices_nifti.plot(cmap=cmap)
