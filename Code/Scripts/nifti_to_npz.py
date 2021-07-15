import os
import gc
import yaml
import numpy as np
import nibabel as nib


base_path = '/mnt/e/HPC_Models/e_field/electrodes'
atlas_path = '/mnt/c/Users/Dimitris/Desktop/Neuro/e_field/rROI_MNI_V7_1mm.nii'
save_path = '/mnt/e/HPC_Models/e_field/electrodes'
# folders = np.sort(next(os.walk(base_path))[1])
folders = ['101309']
e_field_size = []

with open(os.path.realpath('/mnt/c/Users/Dimitris/Documents/repos/ttis-software/Code/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

atlas_nifti = nib.load(atlas_path)
aal_regions_loc = np.where(atlas_nifti.get_fdata().flatten() > 0)[0]
aal_regions_ids = atlas_nifti.get_fdata().flatten()[aal_regions_loc]

settings_electrodes = settings['SfePy']['electrodes']['10-10-mod'].items()
electrode_constant = 10

for fld in folders:
    e_field_values = []
    model_id = fld
    print(model_id)
    
    # electrodes = np.sort(next(os.walk(os.path.join(base_path, fld)))[1])
    
    for electrode in settings_electrodes:
        electrode_name = electrode[0]
        electrode_id = electrode[1]['id'] - electrode_constant
        print(electrode_name)
        files = np.sort(next(os.walk(os.path.join(base_path, fld, electrode_name)))[2])
        
        nifti_images_dict = {}
        for fl in files:
            if not fl.startswith('w'):
                continue
            field_montage = fl.split('.')[0].split('_')[2]
            field_direction = fl.split('.')[0].split('_')[1]
            dict_key = '{}_{}'.format(field_montage, field_direction)
            
            nifti_images_dict[dict_key] = nib.load(os.path.join(base_path, model_id, electrode_name, fl))
        
        e_field = np.nan_to_num(np.vstack((nifti_images_dict[electrode_name + '_x'].get_fdata().flatten(), nifti_images_dict[electrode_name + '_y'].get_fdata().flatten(), nifti_images_dict[electrode_name + '_z'].get_fdata().flatten())).transpose())
        e_field = e_field[aal_regions_loc].reshape((1, -1, 3))
        
        # e_field_size.append(e_field.size)
        
        if isinstance(e_field_values, list):
            e_field_values = e_field
        #     e_field_non_zer_ids = non_zero_pairs
        else:
            e_field_values = np.append(e_field_values, e_field, axis=0)
        #     e_field_non_zer_ids = np.append(e_field_non_zer_ids, non_zero_pairs, axis=0)
        
        del nifti_images_dict
        del e_field
        gc.collect()

    np.savez_compressed(os.path.join(save_path, model_id + '_fields_brain_reg'), e_field=e_field_values.reshape((61, -1, 3)), aal_ids=aal_regions_ids, aal_loc=aal_regions_loc)

    del e_field_values
    gc.collect()
