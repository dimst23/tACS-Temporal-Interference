from __future__ import absolute_import
import importlib
import os
import gc
import sys
from threading import current_thread
import yaml
import numpy as np

with open(os.path.realpath('/mnt/d/Neuro Publication/sim_settings.yml')) as stream:
    settings = yaml.safe_load(stream)

sys.path.append(os.path.realpath(settings['SfePy']['lib_path']))

import FEM.Solver as slv

potential = 1.0
base_path = '/mnt/d/Neuro Publication/'
output_dir = '/mnt/d/Neuro Publication/'
fl = 'meshed_model_sphere.1.vtk'


model = fl.split('_')[-1].split('.')[0]
print(fl)

settings['SfePy']['sphere']['mesh_file'] = os.path.join(base_path, fl)
settings['SfePy']['sphere']['mesh_file_windows'] = os.path.join(base_path, fl)

solve = slv.Solver(settings, 'SfePy', 'sphere')
solve.load_mesh('sphere')
solve.define_field_variable('potential_1', 'voltage_1')
solve.define_field_variable('potential_2', 'voltage_2')
solve.define_field_variable('potential_3', 'voltage_3')
solve.define_field_variable('potential_4', 'voltage_4')


solve.define_essential_boundary('VCC_1', 16, 'potential_1', current=1)
solve.define_essential_boundary('GND_1', 17, 'potential_1', current=-1)
solve.define_essential_boundary('VCC_2', 12, 'potential_2', current=1)
solve.define_essential_boundary('GND_2', 13, 'potential_2', current=-1)

solve.define_essential_boundary('VCC_3', 15, 'potential_3', current=1)
solve.define_essential_boundary('GND_3', 14, 'potential_3', current=-1)
solve.define_essential_boundary('VCC_4', 10, 'potential_4', current=1)
solve.define_essential_boundary('GND_4', 11, 'potential_4', current=-1)



solve.solver_setup(600, 1e-12, 1e-12, verbose=True)
solution = solve.run_solver(save_results=True, post_process_calculation=True, output_dir=output_dir, output_file_name='FEM_' + model)
solve.clear_all()
del solve
gc.collect()
