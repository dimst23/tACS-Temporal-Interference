from __future__ import absolute_import
import os
import gc
import sys
import yaml
import numpy as np
import pandas as pd

with open(os.path.realpath('/home/dimitris/repos/tacs-temporal-interference/Scripts/FEM/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

sys.path.append(os.path.realpath(settings['SfePy']['lib_path']))

import FEM.Solver as slv

# base_path = '/mnt/d/Neuro Publication/'
# output_dir = '/mnt/d/Neuro Publication/'
base_path = '/home/dimitris/Documents/Thesis/Models_with_Electrodes/meshed'
fl = 'Meshed_AAL_101309_10-10_.vtk'

settings['SfePy']['real_brain']['mesh_file'] = os.path.join(base_path, fl)
settings['SfePy']['real_brain']['mesh_file_windows'] = os.path.join(base_path, fl)

electrodes = settings['SfePy']['electrodes']['10-10-mod']
e_field_values = pd.DataFrame()

solve = slv.Solver(settings, 'SfePy', '10-10-mod')
solve.load_mesh('real_brain')

for electrode in electrodes.items():
    if electrode[0] == 'P9':
        continue
    solve.essential_boundaries.clear()
    solve.fields.clear()
    solve.field_variables.clear()

    solve.define_field_variable('potential', 'voltage')

    solve.define_essential_boundary(electrode[0], electrode[1]['id'], 'potential', current=1)
    solve.define_essential_boundary('P9', 71, 'potential', current=-1)

    solve.solver_setup(600, 1e-12, 5e-12, verbose=True)
    solution = solve.run_solver(save_results=False, post_process_calculation=True)

    e_field_base = solution['e_field_(potential)'].data[:, 0, :, 0]
    e_field_values[electrode[0]] = e_field_base

    del solution
    gc.collect()

e_field_values.to_csv(os.path.join(base_path, '101309_fields.csv'))