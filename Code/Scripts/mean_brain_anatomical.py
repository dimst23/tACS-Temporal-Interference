import os
import gc
import numpy as np
import nibabel as nib

# base_path = r'C:\Users\Dimitris\Desktop\Neuro\e_field\affine'
base_path = '/mnt/d/Neuro Publication/All_models_meshed/Nifti/Anatomical/MNI Coordinates'

files = next(os.walk(base_path))[2]
first = True

for fl in files:
    model_id = fl.split('.')[0].split('_')[-1]
    print(model_id)

    nifti_image = nib.load(os.path.join(base_path, fl))
    areas = nifti_image.get_fdata()
    
    if first:
        sum_areas = areas
        affine = nifti_image.affine
        first = False
    else:
        sum_areas += areas

    del areas
    del nifti_image
    gc.collect()


img_header = nib.Nifti1Header()
img_header.set_xyzt_units('mm', 'sec')

new_img = nib.Nifti1Image(sum_areas/len(files), affine, img_header)
nib.save(new_img, os.path.join(base_path, 'Mean_brain_anatomical.nii'))
