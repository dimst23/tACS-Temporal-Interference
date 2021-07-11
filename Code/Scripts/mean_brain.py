import os
import gc
import numpy as np
import nibabel as nib

def modulation_envelope(e_field_1, e_field_2):
    envelope = np.zeros(e_field_1.shape[0])
    # Calculate the angles between the two fields for each vector
    dot_angle = np.einsum('ij,ij->i', e_field_1, e_field_2)
    cross_angle = np.linalg.norm(np.cross(e_field_1, e_field_2), axis=1)
    angles = np.arctan2(cross_angle, dot_angle)
    # Flip the direction of the electric field if the angle between the two is greater or equal to 90 degrees
    e_field_2 = np.where(np.broadcast_to(angles >= np.pi/2., (3, e_field_2.shape[0])).T, -e_field_2, e_field_2)
    # Recalculate the angles
    dot_angle = np.einsum('ij,ij->i', e_field_1, e_field_2)
    cross_angle = np.linalg.norm(np.cross(e_field_1, e_field_2), axis=1)
    angles = np.arctan2(cross_angle, dot_angle)
    E_minus = np.subtract(e_field_1, e_field_2) # Create the difference of the E fields
    # Condition to have two times the E2 field amplitude
    max_condition_1 = np.linalg.norm(e_field_2, axis=1) < np.linalg.norm(e_field_1, axis=1)*np.cos(angles)
    e1_gr_e2 = np.where(np.linalg.norm(e_field_1, axis=1) > np.linalg.norm(e_field_2, axis=1), max_condition_1, False)
    # Condition to have two times the E1 field amplitude
    max_condition_2 = np.linalg.norm(e_field_1, axis=1) < np.linalg.norm(e_field_2, axis=1)*np.cos(angles)
    e2_gr_e1 = np.where(np.linalg.norm(e_field_2, axis=1) > np.linalg.norm(e_field_1, axis=1), max_condition_2, False)
    # Double magnitudes
    envelope = np.where(e1_gr_e2, 2.0*np.linalg.norm(e_field_2, axis=1), envelope) # 2E2 (First case)
    envelope = np.where(e2_gr_e1, 2.0*np.linalg.norm(e_field_1, axis=1), envelope) # 2E1 (Second case)
    # Calculate the complement area to the previous calculation
    e1_gr_e2 = np.where(np.linalg.norm(e_field_1, axis=1) > np.linalg.norm(e_field_2, axis=1), np.logical_not(max_condition_1), False)
    e2_gr_e1 = np.where(np.linalg.norm(e_field_2, axis=1) > np.linalg.norm(e_field_1, axis=1), np.logical_not(max_condition_2), False)
    # Cross product
    envelope = np.where(e1_gr_e2, 2.0*(np.linalg.norm(np.cross(e_field_2, E_minus), axis=1)/np.linalg.norm(E_minus, axis=1)), envelope) # (First case)
    envelope = np.where(e2_gr_e1, 2.0*(np.linalg.norm(np.cross(e_field_1, -E_minus), axis=1)/np.linalg.norm(-E_minus, axis=1)), envelope) # (Second case)
    return np.nan_to_num(envelope)


# base_path = r'C:\Users\Dimitris\Desktop\Neuro\e_field\affine'
base_path = r'/mnt/c/Users/Dimitris/Desktop/Neuro/e_field/affine'

folders = next(os.walk(base_path))[1]
first = True

for fld in folders:
    model_id = fld
    print(model_id)

    files = next(os.walk(os.path.join(base_path, fld)))[2]

    nifti_images_dict = {}
    for fl in files:
        # A naming format like 'wEfield_base_x_101309.nii' is assumed
        if not fl.startswith('w'):
            continue
        field_montage = fl.split('.')[0].split('_')[1]
        field_direction = fl.split('.')[0].split('_')[2]
        dict_key = '{}_{}'.format(field_montage, field_direction)

        nifti_images_dict[dict_key] = nib.load(os.path.join(base_path, model_id, fl))
    
    base_all = 1.25*np.nan_to_num(np.vstack((nifti_images_dict['base_x'].get_fdata().flatten(), nifti_images_dict['base_y'].get_fdata().flatten(), nifti_images_dict['base_z'].get_fdata().flatten())).transpose())
    df_all = 0.75*np.nan_to_num(np.vstack((nifti_images_dict['df_x'].get_fdata().flatten(), nifti_images_dict['df_y'].get_fdata().flatten(), nifti_images_dict['df_z'].get_fdata().flatten())).transpose())
    
    if first:
        sum_e_field_base = base_all
        sum_e_field_df = df_all
        affine = nifti_images_dict['base_x'].affine
        image_size = nifti_images_dict['base_x'].shape
        first = False
    else:
        sum_e_field_base += base_all
        sum_e_field_df += df_all

    del nifti_images_dict
    del base_all
    del df_all
    gc.collect()

    
max_mod = modulation_envelope(sum_e_field_base/len(folders), sum_e_field_df/len(folders))

img_header = nib.Nifti1Header()
img_header.set_xyzt_units('mm', 'sec')

new_img = nib.Nifti1Image(max_mod.reshape(image_size), affine, img_header)
nib.save(new_img, os.path.join(base_path, 'Mean_brain.nii'))
