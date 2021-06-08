from __future__ import absolute_import
import os
import gc
import sys
import yaml
import numpy as np
import pandas as pd

from argparse import ArgumentParser, RawDescriptionHelpFormatter

#### Argument parsing
helps = {
    'settings-file' : "File having the settings to be loaded",
    'model' : "Name of the model. Selection from the settings file",
}

parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument('--version', action='version', version='%(prog)s')
parser.add_argument('--settings-file', metavar='str', type=str,
                    action='store', dest='settings_file',
                    default=None, help=helps['settings-file'], required=True)
parser.add_argument('--meshf', metavar='str', type=str,
                    action='store', dest='meshf',
                    default=None, required=True)
parser.add_argument('--model', metavar='str', type=str,
                    action='store', dest='model',
                    default='real_brain', help=helps['model'], required=True)
parser.add_argument('--csv_save_dir', metavar='str', type=str,
                    action='store', dest='csv_save_dir',
                    default=None, help=helps['model_dir'], required=False)
parser.add_argument('--job_id', metavar='str', type=str,
                    action='store', dest='job_id',
                    default='', required=False)
options = parser.parse_args()
#### Argument parsing

with open(os.path.realpath(options.settings_file)) as stream:
    settings = yaml.safe_load(stream)

if os.name == 'nt':
    extra_path = '_windows'
else:
    extra_path = ''

sys.path.append(os.path.realpath(settings['SfePy']['lib_path' + extra_path]))

import FEM.Solver as slv

settings['SfePy'][options.model]['mesh_file' + extra_path] = options.meshf


electrodes = settings['SfePy']['electrodes']['10-10-mod']
e_field_values = pd.DataFrame()

solve = slv.Solver(settings, 'SfePy', '10-10-mod')
solve.load_mesh(options.model)

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

e_field_values.to_csv(os.path.join(options.csv_save_dir, '101309_fields.csv'))