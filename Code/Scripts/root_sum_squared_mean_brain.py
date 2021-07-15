import os
import gc
import numpy as np
import nibabel as nib

base_path = r'C:\Users\Dimitris\Desktop\Neuro\e_field'
folders = next(os.walk(base_path))[1]

root_sum_squared_models = {}

for fld in folders:
    model_id = fld
    print(model_id)
    #
    files = next(os.walk(os.path.join(base_path, fld)))[2]
    #
    nifti_images_dict = {}
    for fl in files:
        # A naming format like 'wEfield_base_x_101309.nii' is assumed
        if not fl.startswith('w'):
            continue
        field_montage = fl.split('.')[0].split('_')[1]
        field_direction = fl.split('.')[0].split('_')[2]
        dict_key = '{}_{}'.format(field_direction, field_montage)
        #
        nifti_images_dict[dict_key] = nib.load(os.path.join(base_path, model_id, fl))
    #
    base_field = np.nan_to_num(np.vstack((nifti_images_dict['base_x'].get_fdata().flatten(), nifti_images_dict['base_y'].get_fdata().flatten(), nifti_images_dict['base_z'].get_fdata().flatten())).transpose())
    df_field = np.nan_to_num(np.vstack((nifti_images_dict['df_x'].get_fdata().flatten(), nifti_images_dict['df_y'].get_fdata().flatten(), nifti_images_dict['df_z'].get_fdata().flatten())).transpose())
    #
    max_mod = modulation_envelope(base_field, df_field)
    root_sum_squared_models[model_id] = np.sqrt(np.sum(np.power(max_mod, 2)))
    #
    del nifti_images_dict
    gc.collect()
