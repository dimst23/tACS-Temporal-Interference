#!/usr/bin/env python

from __future__ import absolute_import
import os
import sys
import yaml

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
					default=None, help=helps['settings-file'])
parser.add_argument('--model', metavar='str', type=str,
					action='store', dest='model',
					default='sphere', help=helps['model'])
options = parser.parse_args()
#### Argument parsing

## Debuging
options.settings_file = r"C:\Users\Dimitris\Nextcloud\Documents\Neuroscience Bachelor Thesis\Public Repository\tacs-temporal-interference\Scripts\FEM\sim_settings.yml"
options.model = 'sphere'
## Debuging

with open(os.path.realpath(options.settings_file)) as stream:
	settings = yaml.safe_load(stream)

if os.name == 'nt':
	extra_path = '_windows'
else:
	extra_path = ''

#sys.path.append(os.path.realpath(settings['SfePy']['lib_path_windows']))
sys.path.append(os.path.realpath(settings['SfePy']['lib_path' + extra_path]))
import Meshing.modulation_envelope as mod_env

import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds, minimize, least_squares, shgo


#### SfePy libraries
from sfepy.base.base import output, IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Integrals, Equation, Equations, Problem, Function)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyUmfpack, PyAMGSolver, ScipySuperLU
#from sfepy.solvers.ls import ScipyUmfpack, PyAMGSolver, ScipySuperLU, PETScKrylovSolver
from sfepy.solvers.nls import Newton, ScipyBroyden

# import sfepy.parallel.parallel as prl
# from sfepy.parallel.evaluate import PETScParallelEvaluator
#### SfePy libraries

def get_conductivity(ts, coors, mode=None, equations=None, term=None, problem=None, conductivities=None):
	"""[summary]

	Args:
		ts ([type]): [description]
		coors ([type]): [description]
		mode ([type], optional): [description]. Defaults to None.
		equations ([type], optional): [description]. Defaults to None.
		term ([type], optional): [description]. Defaults to None.
		problem ([type], optional): [description]. Defaults to None.
		conductivities ([type], optional): [description]. Defaults to None.

	Returns:
		[type]: [description]
	"""
	# Execute only once at the initialization
	if mode == 'qp':
		values = np.empty(int(coors.shape[0]/4)) # Each element corresponds to one coordinate of the respective tetrahedral edge

		# Save the conductivity values
		for domain in problem.domain.regions:
			if domain.name in conductivities.keys():
				values[domain.entities[3]] = conductivities[domain.name]

		values = np.repeat(values, 4) # Account for the tetrahedral edges
		values.shape = (coors.shape[0], 1, 1)
		
		return {'val' : values}

def post_process(out, problem, state, extend=False):
	from sfepy.base.base import Struct

	e_field_base = problem.evaluate('-ev_grad.2.Omega(potential_base)', mode='qp')
	e_field_df = problem.evaluate('-ev_grad.2.Omega(potential_df)', mode='qp')

	# Calculate the maximum modulation envelope
	modulation_cells = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0])
	modulation_cells = np.repeat(modulation_cells, 4, axis=0).reshape((e_field_base.shape[0], 4, 1, 1))
	modulation_points = modulation_cells.flatten()[np.unique(problem.domain.mesh.get_conn('3_4').flatten(), return_index=True)[1]]

	# Calculate the directional modulation envelope
	modulation_x = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0], dir_vector=[1, 0, 0])
	modulation_y = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0], dir_vector=[0, 1, 0])
	modulation_z = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0], dir_vector=[0, 0, 1])

	modulation_x = np.repeat(modulation_x, 4, axis=0).reshape((e_field_base.shape[0], 4, 1, 1))
	modulation_y = np.repeat(modulation_y, 4, axis=0).reshape((e_field_base.shape[0], 4, 1, 1))
	modulation_z = np.repeat(modulation_z, 4, axis=0).reshape((e_field_base.shape[0], 4, 1, 1))

	# Save the output
	out['e_field_base'] = Struct(name='e_field_base', mode='cell', data=e_field_base, dofs=None)
	out['e_field_df'] = Struct(name='e_field_df', mode='cell', data=e_field_df, dofs=None)
	out['max_modulation'] = Struct(name='max_modulation', mode='cell', data=modulation_cells, dofs=None)
	out['max_modulation_pts'] = Struct(name='max_modulation_pts', mode='vertex', data=modulation_points, dofs=None)
	out['modulation_x'] = Struct(name='modulation_x', mode='cell', data=modulation_x, dofs=None)
	out['modulation_y'] = Struct(name='modulation_y', mode='cell', data=modulation_y, dofs=None)
	out['modulation_z'] = Struct(name='modulation_z', mode='cell', data=modulation_z, dofs=None)

	return out


#mesh = Mesh.from_file(os.path.realpath(settings['SfePy'][options.model]['mesh_file_windows']))
mesh = Mesh.from_file(os.path.realpath(settings['SfePy'][options.model]['mesh_file' + extra_path]))
domain = FEDomain('domain', mesh)

conductivities = {} # Empty conductivity dictionaries

#### Region definition
overall_volume = domain.create_region('Omega', 'all')

for region in settings['SfePy'][options.model]['regions'].items():
	domain.create_region(region[0], 'cells of group ' + str(region[1]['id']))
	conductivities[region[0]] = region[1]['conductivity']

for electrode in settings['SfePy'][options.model]['electrodes'].items():
	if electrode[0] != 'conductivity':
		domain.create_region(electrode[0], 'cells of group ' + str(electrode[1]['id']))
		conductivities[electrode[0]] = settings['SfePy'][options.model]['electrodes']['conductivity']
#### Region definition

#### Material definition
conductivity = Material('conductivity', function=Function('get_conductivity', lambda ts, coors, mode=None, equations=None, term=None, problem=None, **kwargs: get_conductivity(ts, coors, mode, equations, term, problem, conductivities=conductivities)))
#conductivity = Material('conductivities', function=Function(domain, settings['SfePy'][options.model]['conductivities']))
#### Material definition

#### Solver definition

ls_status = IndexedStruct()

ls = PyAMGSolver({
	'i_max': 100,
	'eps_r': 1e-12,
}, status=ls_status)

#ls = ScipyUmfpack({}, status=ls_status)

nls_status = IndexedStruct()
nls = Newton({
	'i_max': 1,
	'eps_a': 1e-4,
}, lin_solver=ls, status=nls_status)
#### Solver definition

r_base_vcc = domain.create_region('Base_VCC', 'vertices of group 4', 'facet')
r_base_gnd = domain.create_region('Base_GND', 'vertices of group 5', 'facet')
r_df_vcc = domain.create_region('DF_VCC', 'vertices of group 6', 'facet')
r_df_gnd = domain.create_region('DF_GND', 'vertices of group 7', 'facet')

def ackley(x):
	arg1 = -0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
	arg2 = 0.5 * (np.cos(2. * np.pi * x[0]) + np.cos(2. * np.pi * x[1]))
	return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e

## Optimization starts here
def electrode_optimization(x, *data):
	domain, conductivity, solver, post_process_evolution = data
	x = np.round(x, 4)
	#### Essential boundary conditions
	bc_base_vcc = EssentialBC('base_vcc', r_base_vcc, {'potential_base.all' : 150.0})
	bc_base_gnd = EssentialBC('base_gnd', r_base_gnd, {'potential_base.all' : -150.0})

	bc_df_vcc = EssentialBC('df_vcc', r_df_vcc, {'potential_df.all' : 150.0*x[0]})
	bc_df_gnd = EssentialBC('df_gnd', r_df_gnd, {'potential_df.all' : -150.0*x[0]})
	#### Essential boundary conditions

	#### Field definition
	field_potential = Field.from_args('voltage', dtype=np.float64, shape=(1, ), region=overall_volume, approx_order=1)

	fld_potential_base = FieldVariable('potential_base', 'unknown', field=field_potential)
	fld_s_base = FieldVariable('s_base', 'test', field=field_potential, primary_var_name='potential_base')

	fld_potential_df = FieldVariable('potential_df', 'unknown', field=field_potential)
	fld_s_df = FieldVariable('s_df', 'test', field=field_potential, primary_var_name='potential_df')
	#### Field definition

	#### Equation definition
	integral = Integral('i1', order=2)

	laplace_base = Term.new('dw_laplace(conductivity.val, s_base, potential_base)', integral, region=overall_volume, conductivity=conductivity, potential_base=fld_potential_base, s_base=fld_s_base)
	laplace_df = Term.new('dw_laplace(conductivity.val, s_df, potential_df)', integral, region=overall_volume, conductivity=conductivity, potential_df=fld_potential_df, s_df=fld_s_df)

	eq_base = Equation('balance', laplace_base)
	eq_df = Equation('balance', laplace_df)

	equations = Equations([eq_base, eq_df])
	#### Equation definition

	problem = Problem('temporal_interference', equations=equations)
	problem.set_bcs(ebcs=Conditions([bc_base_vcc, bc_base_gnd, bc_df_vcc, bc_df_gnd]))
	problem.set_solver(solver)

	# Solve the problem
	state = problem.solve(post_process_hook=post_process)
	#state = problem.solve()
	#output(ls_status)
	#output(nls_status)
	#problem.evaluate()

	#output_data = state.create_output_dict()
	e_field_base = problem.evaluate('-ev_grad.2.Omega(potential_base)', mode='qp')
	e_field_df = problem.evaluate('-ev_grad.2.Omega(potential_df)', mode='qp')

	modulation_cells = mod_env.modulation_envelope(e_field_base[:, 0, :, 0], e_field_df[:, 0, :, 0])
	modulation_cells = np.round(modulation_cells, 6)
	del state
	del e_field_base
	del e_field_df

	#problem.save_state('file.vtk', out=output_data)
	# mod_val = round(modulation_cells[278668], 5)
	print("Variables")
	print(x[0])

	crds = problem.domain.mesh.coors

	bounding_roi = {
		'x_min': 47.0,
		'x_max': 59.0,
		'y_min': 10.0,
		'y_max': 27.0,
		'z_min': -14.0,
		'z_max': -3.0
	}

	vert_x = np.logical_and(crds[:, 0] >= bounding_roi['x_min'], crds[:, 0] <= bounding_roi['x_max'])
	vert_y = np.logical_and(crds[:, 1] >= bounding_roi['y_min'], crds[:, 1] <= bounding_roi['y_max'])
	vert_z = np.logical_and(crds[:, 2] >= bounding_roi['z_min'], crds[:, 2] <= bounding_roi['z_max'])
	
	# Get the ROI vertex indices
	vert_id_roi = np.arange(crds.shape[0])
	roi_ids = (vert_x * vert_y * vert_z > 0)
	# roi_ids = (vert_x * vert_y > 0)

	common_points = np.isin(problem.domain.mesh.get_conn('3_4'), vert_id_roi[roi_ids])
	common_points = np.where(np.sum(common_points, axis=1) == 4)[0]
	
	mod_val = round(np.average(modulation_cells[common_points]), 6)

	print(mod_val)
	print(np.amax(modulation_cells[common_points]))
	return [-mod_val, -np.amax(modulation_cells[common_points])]

#nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)
#bounds = Bounds([0, 0], [500, 500])
bounds = [(1e-4, 5.)]
# bounds_lsq = ([1e-4], [5.])
#result = differential_evolution(electrode_optimization, bounds, args=(domain, conductivity, nls, post_process), constraints=(nlc), seed=1, maxiter=4)
result = differential_evolution(electrode_optimization, bounds, args=(domain, conductivity, nls, post_process), disp=True)

# result_min = minimize(electrode_optimization, np.array([1,]), args=(domain, conductivity, nls, post_process), bounds=bounds)
# result_sq = least_squares(electrode_optimization, np.array([1.,]), args=(domain, conductivity, nls, post_process), bounds=bounds_lsq)
# result_shgo = shgo(electrode_optimization, bounds, args=(domain, conductivity, nls, post_process))
