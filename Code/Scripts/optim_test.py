from __future__ import absolute_import
import importlib
import os
import gc
import sys
import yaml
import numpy as np

with open(os.path.realpath('/home/dimitris/repos/tacs-temporal-interference/Scripts/FEM/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

sys.path.append(os.path.realpath(settings['SfePy']['lib_path']))

import FEM.Optimization as optm

base_path = '/home/dimitris/Documents/Thesis/Models_with_Electrodes/meshed'
fl = 'Meshed_AAL_101309_10-10_.vtk'

settings['SfePy']['real_brain']['mesh_file'] = os.path.join(base_path, fl)
settings['SfePy']['real_brain']['mesh_file_windows'] = os.path.join(base_path, fl)

# electrodes_to_omit = ['Nz', 'N2', 'AF10', 'F10', 'FT10', 'T10(M2)', 'TP10', 'PO10', 'I2', 'Iz', 'I1', 'PO9', 'TP9', 'T9(M1)', 'FT9', 'F9', 'AF9', 'F1', 'P9', 'P10']
# for elec in electrodes_to_omit:
#     settings['SfePy']['electrodes']['10-10-mod'].pop(elec)

optimization = optm.Optimization(settings, 'SfePy', '10-10-mod')
optimization.initialization('real_brain', 600, 1e-12, 5e-12)
optz = optimization.run_optimization()

