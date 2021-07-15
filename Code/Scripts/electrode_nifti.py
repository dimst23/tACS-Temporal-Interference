import os
import gc
import numpy as np
import nibabel as nib


base_path = r'E:\HPC_Models\e_field\electrodes'
atlas_path = r'C:\Users\Dimitris\Desktop\Neuro\e_field\rROI_MNI_V7_1mm.nii'
folders = np.sort(next(os.walk(base_path))[1])
# folders = ['101309']
ommit_models = ['']
e_field_values = []
e_field_non_zer_ids = []
e_field_size = []

atlas_nifti = nib.load(atlas_path)
aal_regions_loc = np.where(atlas_nifti > 0)[0]

for fld in folders:
    if fld in ommit_models:
        continue
    model_id = fld
    print(model_id)
    
    electrodes = next(os.walk(os.path.join(base_path, fld)))[1]
    
    for electrode in electrodes:
        print(electrode)
        files = next(os.walk(os.path.join(base_path, fld, electrode)))[2]
        
        nifti_images_dict = {}
        for fl in files:
            if not fl.startswith('w'):
                continue
            field_montage = fl.split('.')[0].split('_')[2]
            field_direction = fl.split('.')[0].split('_')[1]
            dict_key = '{}_{}'.format(field_montage, field_direction)
            
            nifti_images_dict[dict_key] = nib.load(os.path.join(base_path, model_id, electrode, fl))
        
        e_field = np.nan_to_num(np.vstack((nifti_images_dict[electrode + '_x'].get_fdata().flatten(), nifti_images_dict[electrode + '_y'].get_fdata().flatten(), nifti_images_dict[electrode + '_z'].get_fdata().flatten())).transpose())
        e_field = e_field[aal_regions_loc]
        
        non_zero_pairs = np.where(np.sum(e_field == 0, axis=1) != 3)[0]
        non_zero_values = e_field[non_zero_pairs].reshape((1, -1, 3))
        e_field_size.append(e_field.size)
        
        if isinstance(e_field_values, list):
            e_field_values = non_zero_values
            e_field_non_zer_ids = non_zero_pairs
        else:
            e_field_values = np.append(e_field_values, non_zero_values, axis=0)
            e_field_non_zer_ids = np.append(e_field_non_zer_ids, non_zero_pairs, axis=0)
        
        del nifti_images_dict
        gc.collect()

